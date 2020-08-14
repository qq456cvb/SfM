import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
from SFM.triangulation import triangulate
from FeatureMatching import compute_matches
import open3d as o3d
from SFM.RANSAC import RANSAC, TrifocalKernel
from scipy.optimize import linprog
from cvxopt import matrix, solvers

if __name__ == "__main__":
    np.random.seed(0)
    
    intrinsic = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
    intrinsic_inv = np.linalg.inv(intrinsic)
    
    num_imgs = 3
    
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
    
    kps1 = descriptors[0][0]
    kps2 = descriptors[1][0]
    corr1 = np.array([kps1[match[0]] for match in matches[(0, 1)]])
    corr2 = np.array([kps2[match[1]] for match in matches[(0, 1)]])
    corr1 = np.concatenate([corr1, np.ones((corr1.shape[0], 1))], -1)
    corr2 = np.concatenate([corr2, np.ones((corr2.shape[0], 1))], -1)
    
    corr1 = (intrinsic_inv @ corr1.T).T
    corr2 = (intrinsic_inv @ corr2.T).T
    corrs = np.concatenate([corr1[:, :2], corr2[:, :2]], -1)
    
    P2 = [
        np.array([[0.450927, -0.0945642, -0.887537 , -7.28137], [-0.892535, -0.0401974, -0.449183 , -7.57667], [0.00679989, 0.994707, -0.102528, 0.204446]]),
        np.array([[0.582226, -0.0983866, -0.807052, -8.31326], [-0.813027, -0.0706383, -0.577925, -6.3181], [-0.000148752, 0.992638, -0.121118, 0.16107]]),
        np.array([[0.666779, -0.0831384, -0.740603, -9.46627], [-0.74495, -0.0459057, -0.665539, -5.58174], [0.021334, 0.99548, -0.0925429, 0.147736 ]]),
    ]
    
    P2[1][:3, -1] -= P2[0][:3, -1]
    P2[2][:3, -1] -= P2[0][:3, -1]
    P2[0][:3, -1] = 0
    for i in range(len(P2)):
        P2[i] = np.concatenate([P2[i][:3,:3].T, -np.dot(P2[i][:3,:3].T, P2[i][:3,-1])[:, None]], -1)
    
    # print(intrinsic_inv)
    
    # print(P2[0])
    # print(P2[1])
    # print(P2[2])
    
    Rs = np.stack([P2[0][:3, :3], P2[1][:3, :3], P2[2][:3, :3]])
    # pts = triangulate(P2[:2], corrs)[:, :-1]
        
    # pred_pcd = o3d.geometry.PointCloud()
    # # Map color
    # colors = np.array([[0.7, 0., 0.] for l in pts])
    # pred_pcd.points = o3d.utility.Vector3dVector(pts)
    # pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # # Visualize the input point cloud and the prediction
    # o3d.visualization.draw_geometries([pred_pcd])
    
    # synthetic dataset
    X = np.concatenate([-np.random.uniform(size=(4, 3)), np.ones((4, 1))], -1)
    # X[:, -2] += 0.5
    
    projection1 = np.einsum('rc,nc->nr', P2[0], X)
    projection2 = np.einsum('rc,nc->nr', P2[1], X)
    projection3 = np.einsum('rc,nc->nr', P2[2], X)
    # print(projection1[:, -1])
    projection1 = projection1[:, :2] / projection1[:, -1:]
    projection2 = projection2[:, :2] / projection2[:, -1:]
    projection3 = projection3[:, :2] / projection3[:, -1:]
    
    image_points_normalized = np.stack([projection1, projection2, projection3], 1)
    # # print(image_points_normalized.shape)
    
    # records = image_points_normalized.reshape(-1, 6)
    # x11, y11, x12, y12, x13, y13 = [records[0, i] for i in range(6)]
    # x21, y21, x22, y22, x23, y23 = [records[1, i] for i in range(6)]
    # x31, y31, x32, y32, x33, y33 = [records[2, i] for i in range(6)]
    # x41, y41, x42, y42, x43, y43 = [records[3, i] for i in range(6)]
    
    # thresh = 0.01
    # # abs
    # A_ub = np.array([
    #     # positive parts
    #     [0, 0, 0, 0, 0, 0, *(Rs[0][2] * (x11 - thresh) - Rs[0][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, *(Rs[0][2] * (y11 - thresh) - Rs[0][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
    #     [-1, 0, x12 - thresh, 0, 0, 0, *(Rs[1][2] * (x12 - thresh) - Rs[1][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, -1, y12 - thresh, 0, 0, 0, *(Rs[1][2] * (y12 - thresh) - Rs[1][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 0, 0, -1, 0, x13 - thresh, *(Rs[2][2] * (x13 - thresh) - Rs[2][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, -1, y13 - thresh, *(Rs[2][2] * (y13 - thresh) - Rs[2][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (x21 - thresh) - Rs[0][0]), 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (y21 - thresh) - Rs[0][1]), 0, 0, 0, 0, 0, 0],  # view 1
    #     [-1, 0, x22 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (x22 - thresh) - Rs[1][0]), 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, -1, y22 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (y22 - thresh) - Rs[1][1]), 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 0, 0, -1, 0, x23 - thresh, 0, 0, 0, *(Rs[2][2] * (x23 - thresh) - Rs[2][0]), 0, 0, 0, 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, -1, y23 - thresh, 0, 0, 0, *(Rs[2][2] * (y23 - thresh) - Rs[2][1]), 0, 0, 0, 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (x31 - thresh) - Rs[0][0]), 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (y31 - thresh) - Rs[0][1]), 0, 0, 0],  # view 1
    #     [-1, 0, x32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (x32 - thresh) - Rs[1][0]), 0, 0, 0],  # view 2
    #     [0, -1, y32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (y32 - thresh) - Rs[1][1]), 0, 0, 0],  # view 2
    #     [0, 0, 0, -1, 0, x33 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (x33 - thresh) - Rs[2][0]), 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, -1, y33 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (y33 - thresh) - Rs[2][1]), 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (x41 - thresh) - Rs[0][0])],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (y41 - thresh) - Rs[0][1])],  # view 1
    #     [-1, 0, x42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (x42 - thresh) - Rs[1][0])],  # view 2
    #     [0, -1, y42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (y42 - thresh) - Rs[1][1])],  # view 2
    #     [0, 0, 0, -1, 0, x43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (x43 - thresh) - Rs[2][0])],  # view 3
    #     [0, 0, 0, 0, -1, y43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (y43 - thresh) - Rs[2][1])],  # view 3
        
    #     # negative parts
    #     [0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-x11 - thresh) + Rs[0][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-y11 - thresh) + Rs[0][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
    #     [1, 0, -x12 - thresh, 0, 0, 0, *(Rs[1][2] * (-x12 - thresh) + Rs[1][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 1, -y12 - thresh, 0, 0, 0, *(Rs[1][2] * (-y12 - thresh) + Rs[1][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 0, 0, 1, 0, -x13 - thresh, *(Rs[2][2] * (-x13 - thresh) + Rs[2][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, 1, -y13 - thresh, *(Rs[2][2] * (-y13 - thresh) + Rs[2][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-x21 - thresh) + Rs[0][0]), 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-y21 - thresh) + Rs[0][1]), 0, 0, 0, 0, 0, 0],  # view 1
    #     [1, 0, -x22 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-x22 - thresh) + Rs[1][0]), 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 1, -y22 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-y22 - thresh) + Rs[1][1]), 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 0, 0, 1, 0, -x23 - thresh, 0, 0, 0, *(Rs[2][2] * (-x23 - thresh) + Rs[2][0]), 0, 0, 0, 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, 1, -y23 - thresh, 0, 0, 0, *(Rs[2][2] * (-y23 - thresh) + Rs[2][1]), 0, 0, 0, 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-x31 - thresh) + Rs[0][0]), 0, 0, 0],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-y31 - thresh) + Rs[0][1]), 0, 0, 0],  # view 1
    #     [1, 0, -x32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-x32 - thresh) + Rs[1][0]), 0, 0, 0],  # view 2
    #     [0, 1, -y32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-y32 - thresh) + Rs[1][1]), 0, 0, 0],  # view 2
    #     [0, 0, 0, 1, 0, -x33 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (-x33 - thresh) + Rs[2][0]), 0, 0, 0],  # view 3
    #     [0, 0, 0, 0, 1, -y33 - thresh, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (-y33 - thresh) + Rs[2][1]), 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-x41 - thresh) + Rs[0][0])],  # view 1
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[0][2] * (-y41 - thresh) + Rs[0][1])],  # view 1
    #     [1, 0, -x42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-x42 - thresh) + Rs[1][0])],  # view 2
    #     [0, 1, -y42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[1][2] * (-y42 - thresh) + Rs[1][1])],  # view 2
    #     [0, 0, 0, 1, 0, -x43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (-x43 - thresh) + Rs[2][0])],  # view 3
    #     [0, 0, 0, 0, 1, -y43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(Rs[2][2] * (-y43 - thresh) + Rs[2][1])],  # view 3
        
    #     # front constraints
    #     [0, 0, 0, 0, 0, 0, *(-Rs[0][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, -1, 0, 0, 0, *(-Rs[1][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
    #     [0, 0, 0, 0, 0, -1, *(-Rs[2][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[0][2]), 0, 0, 0, 0, 0, 0],  # view 1
    #     [0, 0, -1, 0, 0, 0, 0, 0, 0, *(-Rs[1][2]), 0, 0, 0, 0, 0, 0], # view 2
    #     [0, 0, 0, 0, 0, -1, 0, 0, 0, *(-Rs[2][2]), 0, 0, 0, 0, 0, 0], # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[0][2]), 0, 0, 0],  # view 1
    #     [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[1][2]), 0, 0, 0],  # view 2
    #     [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, *(-Rs[2][2]), 0, 0, 0],  # view 3
        
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[0][2])], # view 1
    #     [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[1][2])],  # view 2
    #     [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-Rs[2][2])]  # view 3
    # ])
    # c = np.zeros((18,))
    # b_ub = np.zeros((60,))
    # b_ub[-12:] = -1.
    
    # variables = np.array([*P2[1][:3, -1], *P2[2][:3, -1], *X[:, :3].reshape(-1)])
    # test = A_ub @ variables
    # # test = test[-12:]
    # print(test)
    # # print(A_ub[24])
    # # print(Rs[0][2])
    # # print(variables)
    # # print(A_ub @ variables <= b_ub)
    # # res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="revised simplex")
    # # print(res.success, res.status)
    
    # sol = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub))
    # opt_x = np.array(sol['x']).reshape(-1)
    # print(sol)
    # print(A_ub @ opt_x)
    # print(opt_x[-12:])
    # print(X[:, :3].reshape(-1))
    
    ransac = RANSAC(TrifocalKernel(1e-1, 4, Rs))
    best_model, best_inlier_idxs = ransac.process(image_points_normalized.reshape(-1, 6))
    
    print(X[:, :3].reshape(-1))