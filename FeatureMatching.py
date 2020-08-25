import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy
from tqdm import tqdm
import open3d as o3d
from SFM.Union import UnionTree
from scipy.optimize import linprog
from SFM.RANSAC import *
from SFM.triangulation import triangulate
from SFM.visualization import draw_cameras
        
        
intrinsic = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
intrinsic_inv = np.linalg.inv(intrinsic)
    
        
def compute_matches(descs1, descs2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(descs1, descs2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return [(match.queryIdx, match.trainIdx) for match in matches]


def get_relative_P(img1, img2, kps1, kps2, matches):
    # import pdb; pdb.set_trace()
    # Initiate SIFT detector
    # orb = cv2.ORB_create()

    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)

    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # Match descriptors.
    # matches = bf.match(des1,des2)

    # # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches, None, flags=2)
    # plt.figure()
    # plt.imshow(img3)
    
    T = np.array([[1. / img1.shape[1], 0, -0.5], [0, 1. / img1.shape[0], -0.5], [0, 0, 1]])
    
    # kp1_idx = np.array([m.queryIdx for m in matches])
    # kp2_idx = np.array([m.trainIdx for m in matches])
    # corr1 = cv2.KeyPoint_convert(kp1)[kp1_idx]
    # corr2 = cv2.KeyPoint_convert(kp2)[kp2_idx]
    corr1 = np.array([kps1[match[0]] for match in matches])
    corr2 = np.array([kps2[match[1]] for match in matches])
    corr1 = np.concatenate([corr1, np.ones((corr1.shape[0], 1))], -1)
    corr2 = np.concatenate([corr2, np.ones((corr2.shape[0], 1))], -1)
    
    corr1 = (T @ corr1.T).T
    corr2 = (T @ corr2.T).T
    
    ransac = RANSAC(HomographyKernel(2.5 / max(img1.shape[0], img1.shape[1]), 8))
    best_model, best_inlier_idxs = ransac.process(np.concatenate([corr1[:, :2], corr2[:, :2]], -1))
    # print(len(best_inlier_idxs), len(matches))
    # print(best_model)
    # print(np.linalg.det(best_model))
    if len(best_inlier_idxs) < 50:
        return None
    
    matches = [matches[i] for i in best_inlier_idxs]
    
    # import pdb; pdb.set_trace()
    # cvkps1 = [cv2.KeyPoint(x=point[0],y=point[1],_size=0, _angle=0,_response=0, _octave=0,_class_id=0) for point in kps1]
    # cvkps2 = [cv2.KeyPoint(x=point[0],y=point[1],_size=0, _angle=0,_response=0, _octave=0,_class_id=0) for point in kps2]
    # cvmatches = [cv2.DMatch(match[0], match[1], 0) for match in matches]
    # img4 = cv2.drawMatches(img1, cvkps1, img2, cvkps2, cvmatches, None, flags=2)
    
    # plt.figure()
    # plt.imshow(img4)
    # plt.show()

    # kp1_idx = np.array([m.queryIdx for m in matches])
    # kp2_idx = np.array([m.trainIdx for m in matches])
    # corr1 = cv2.KeyPoint_convert(kp1)[kp1_idx]
    # corr2 = cv2.KeyPoint_convert(kp2)[kp2_idx]
    corr1 = np.array([kps1[match[0]] for match in matches])
    corr2 = np.array([kps2[match[1]] for match in matches])
    corr1 = np.concatenate([corr1, np.ones((corr1.shape[0], 1))], -1)
    corr2 = np.concatenate([corr2, np.ones((corr2.shape[0], 1))], -1)
    
    corr1 = (intrinsic_inv @ corr1.T).T
    corr2 = (intrinsic_inv @ corr2.T).T
    
    F = T.T @ best_model @ T
    
    E = intrinsic.T @ F @ intrinsic  # TODO: direct compute essential from unnormalized 3D points
    
    u, s, vh = np.linalg.svd(E)
    # s = np.array([(s[0] + s[1]) / 2, (s[0] + s[1]) / 2, 0])  # not necessary since we will svd E later
    # E = u @ np.diag(s) @ vh
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    corrs = np.concatenate([corr1[:, :2], corr2[:, :2]], -1)
    
    valid_P = None
    best_cnt = 0
    for w in [W, W.T]:
        for t in [u[:, -1], -u[:, -1]]:
            rot = u @ w @ vh
            translation = t[:, None]
            if np.linalg.det(rot) < 0:
                rot = -rot
                translation = -translation
            P = np.concatenate([rot, translation], -1)
            
            cnt = 0
            
            P2 = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), P]
            Xs = triangulate(P2, corrs)
            for X in Xs:
                PX = np.dot(P, X)
                
                if X[-2] > 0 and PX[-1] > 0:
                    cnt += 1
            # for c in corrs:
            #     X = triangulate(P, c)

            #     PX = np.dot(P, X)
            #     if X[-2] > 0 and PX[-1] > 0:
            #         cnt += 1
            # print(cnt)
            if cnt > best_cnt:
                best_cnt = cnt
                valid_P = P
    # print(best_cnt, corrs.shape[0])
            
    P2 = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), valid_P]
    pts = triangulate(P2, corrs)[:, :-1]
    pts[:, 2] = -pts[:, 2]  # open3d is right hand coordinate
    # pts = []
    # for c in corrs:
    #     pts.append(triangulate(valid_P, c)[:-1])
    # pts = np.stack(pts)
    # print(pts.shape)
        
    pred_pcd = o3d.geometry.PointCloud()
    # Map color
    colors = np.array([[0.7, 0., 0.] for l in pts])
    pred_pcd.points = o3d.utility.Vector3dVector(pts)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # Visualize the input point cloud and the prediction
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pred_pcd, mesh_frame])
    return valid_P, matches


def get_triplet_matches(triplet, matches):
    i, j, k = triplet
    all_feats = list(set([(i, match[0]) for match in matches[(i, j)]] + [(j, match[1]) for match in matches[(i, j)]] + [(j, match[0]) for match in matches[(j, k)]] + [(k, match[1]) for match in matches[(j, k)]]))
    # print(len(matches[(i, j)]), len(matches[(j, k)]), len(all_feats))
    
    feat2idx = dict([(feat, idx) for idx, feat in enumerate(all_feats)])
    
    ut = UnionTree(len(all_feats))
    
    for match in matches[(i, j)]:
        ut.union(feat2idx[(i, match[0])], feat2idx[(j, match[1])])
        
    for match in matches[(j, k)]:
        ut.union(feat2idx[(j, match[0])], feat2idx[(k, match[1])])
        
    triplet_matches = {}
    for feat in all_feats:
        idx = ut.find(feat2idx[feat])
        if idx in triplet_matches:
            triplet_matches[idx].append(feat)
        else:
            triplet_matches[idx] = [feat]
    
    triplet_matches = {k: v for k, v in triplet_matches.items() if len(v) == 3}
    return triplet_matches
    


if __name__ == "__main__":
    # img1 = cv2.imread('Benchmarking_Camera_Calibration_2008/fountain-P11/images/0000.jpg', 0) # queryImage
    # img2 = cv2.imread('Benchmarking_Camera_Calibration_2008/fountain-P11/images/0001.jpg', 0) # trainImage
    num_imgs = 4
    
    imgs = [cv2.imread(img, 0) for img in glob('Benchmarking_Camera_Calibration_2008/fountain-P11/images/*.jpg')]
    
    orb = cv2.ORB_create(nfeatures=2000)
    descriptors = {}
    for i, img in enumerate(imgs):
        kps, descs = orb.detectAndCompute(img,None)
        descriptors[i] = (cv2.KeyPoint_convert(kps), descs)
    
    matches = {}
    adjs = {}
    for i in range(num_imgs):
        adjs[i] = set()
        for j in range(i+1, num_imgs):
            matches[(i, j)] = compute_matches(descriptors[i][1], descriptors[j][1])
            if len(matches[(i, j)]) > 50:
                adjs[i].add(j)
                
    triplets = []
    for i in range(num_imgs):
        for j in range(i+1, num_imgs):
            intersection = adjs[i].intersection(adjs[j])
            for k in intersection:
                triplets.append((i, j, k))

    relative_rotations = {}
    A = []
    Rijs = np.zeros((num_imgs, num_imgs, 3, 3))
    for i in tqdm(range(num_imgs)):
        for j in range(i+1, num_imgs):
            Pij, matches[(i, j)] = get_relative_P(imgs[i], imgs[j], descriptors[i][0], descriptors[j][0], matches[(i, j)])
            if Pij is not None:
                Rij = Pij[:3, :3]
                relative_rotations[(i, j)] = Rij
                Rijs[i][j] = Rij
                row = np.zeros((3, num_imgs * 3))
                row[:, i*3:(i+1)*3] = Rij
                row[:, j*3:(j+1)*3] = -np.identity(3)
                A.append(row)
    A = np.concatenate(A, 0)
    # _, _, vh = np.linalg.svd(A)
    # print(A @ vh[-3:].T)
    # _, R_concat = eigsh(-A.T @ A, 3)
    _, R_concat = scipy.linalg.eigh(A.T @ A, eigvals=(0, 2))
    # print(A @ R_concat)
    # R_concat = vh[-3:].T
    Rs = []
    for i in range(num_imgs):
        R = R_concat[i*3:(i+1)*3]
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        Rs.append(R)
    
    Rs = np.stack(Rs)
        
    all_triplet_matches = {}
    for triplet in triplets:
        # triplet_matches = get_triplet_matches(triplet, matches)
        all_triplet_matches[triplet] = list(get_triplet_matches(triplet, matches).values())
        # print(triplet_matches)
    #     print(len(triplet_matches.values()))
        
    # TODO: use normalized coord
    relative_translations = {}
    for triplet, triplet_matches in all_triplet_matches.items():
        image_points = np.stack([np.stack([descriptors[i][0][j] for (i, j) in triplet]) for triplet in triplet_matches])
        image_points = np.concatenate([image_points, np.ones((image_points.shape[0], image_points.shape[1], 1))], -1)
        image_points_normalized = np.einsum('rc,nvc->nvr', intrinsic_inv, image_points)
        ransac = RANSAC(TrifocalKernel(1e-2, 4, Rs))
        best_model, best_inlier_idxs = ransac.process(image_points_normalized[:, :, :2].reshape(-1, 6))
        # print(best_model, len(best_inlier_idxs))
        
        i, j, k = triplet
        t2 = best_model[:3]
        t3 = best_model[3:]
        relative_translations[(i, j)] = t2
        relative_translations[(i, k)] = t3
        relative_translations[(j, k)] = t3 - np.dot(relative_rotations[(j, k)], t2)
    
    triplet_cnt = len(all_triplet_matches)
    A_ub = np.zeros((19 * triplet_cnt, num_imgs * 3 + triplet_cnt + 1))
    cnt = 0
    for idx, triplet in enumerate(all_triplet_matches):
        i, j, k = triplet
        for pair in [(i, j), (i, k), (j, k)]:
            a, b = pair
            A_ub[cnt:cnt + 3, a * 3:(a + 1) * 3] = -relative_rotations[(a, b)]
            A_ub[cnt:cnt + 3, b * 3:(b + 1) * 3] = np.eye(3)
            A_ub[cnt:cnt + 3, num_imgs * 3 + idx] = -relative_translations[(a, b)]
            A_ub[cnt:cnt + 3, -1] = -1
            cnt += 3
        
        for pair in  [(i, j), (i, k), (j, k)]:
            a, b = pair
            A_ub[cnt:cnt + 3, a * 3:(a + 1) * 3] = relative_rotations[(a, b)]
            A_ub[cnt:cnt + 3, b * 3:(b + 1) * 3] = -np.eye(3)
            A_ub[cnt:cnt + 3, num_imgs * 3 + idx] = relative_translations[(a, b)]
            A_ub[cnt:cnt + 3, -1] = -1
            cnt += 3
    
    A_ub[-triplet_cnt:, num_imgs * 3:num_imgs * 3 + triplet_cnt] = -np.eye(triplet_cnt)
    # print(cnt, 18 * triplet_cnt)
    # T1 = 0
    A_ub = A_ub[:, 3:]
    b_ub = np.zeros((A_ub.shape[0],))
    b_ub[-triplet_cnt:] = -1
    c = np.zeros((num_imgs * 3 + triplet_cnt + 1 - 3,))
    c[-1] = 1.
    # print(A_ub[-triplet_cnt:])
    res = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub))
    opt = np.array(res['x']).reshape(-1)
    
    print(A_ub @ opt)
    
    print('best error', opt[-1])
    print('tau', opt[(num_imgs - 1) * 3:-1])
    translations = opt[:(num_imgs - 1) * 3].reshape(-1, 3)
    translations = np.concatenate([np.zeros((1, 3)), translations], 0)
    print('translations', translations)
    
    Ps = np.concatenate([Rs, translations[:, :, None]], -1)
    draw_cameras(Ps)
    # print(np.arccos((np.trace(Rs[0] @ Rijs[0][1] @ Rs[1].T) - 1) / 2) / np.pi * 180)