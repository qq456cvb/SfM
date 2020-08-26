# import taichi as ti
import cv2
import numpy as np
# from cupyx.scipy import ndimage
from SFM.utils import read_camera_extrinsic
from SFM.triangulation import compute_F_from_P, opt_triangulate
from glob import glob
from scipy import optimize, signal
from tqdm import tqdm
import open3d as o3d
from patch import Cell, Patch
from utils import *

                    
# gray_convert = np.RawKernel(r'''
#     extern "C" __global__ void to_grayscale(const float* input, float* output, int width, int height) {
#             int x = blockDim.x * blockIdx.x + threadIdx.x;
#             int y = blockDim.y * blockIdx.y + threadIdx.y;
#             int idx = y * width + x;
#             if (x > width - 1 || y > height - 1) return;
#             output[idx] = 0.287 * input[idx * 3] + 0.587 * input[idx * 3 + 1] + 0.114 * input[idx * 3 + 2];
#     }
# ''', 'to_grayscale'
# )

# gradient_compute = np.RawKernel(r'''
#     extern "C" __global__ void gradient(const float* input, float* output, int width, int height, int direction) {
#             int x = blockDim.x * blockIdx.x + threadIdx.x;
#             int y = blockDim.y * blockIdx.y + threadIdx.y;
#             int idx = y * width + x;
#             if (x < 1 || x > width - 2 || y< 1 || y > height - 2) return;
#             if (direction == 0) {
#                 output[idx] = (input[y * width + x + 1] - input[y * width + x - 1]) / 2;
#             } else {
#                 output[idx] = (input[(y + 1) * width + x] - input[(y - 1) * width + x]) / 2;
#             }
#     }
# ''', 'gradient'
# )





def get_gaussian_kernel(sigma=1.):
    radius = int(3 * sigma) + 1
    xv, yv = np.meshgrid(np.arange(radius * 2 + 1), np.arange(radius * 2 + 1))
    xv -= radius
    yv -= radius
    kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def get_feature_points(harris_response, DoG):
    feature_points = set()
    for i in range((DoG.shape[0] + 31) // 32):
        for j in range((DoG.shape[1] + 31) // 32):
            block_harris = harris_response[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
            max_idx = np.stack(np.unravel_index(np.argsort(np.ravel(-block_harris))[:4], block_harris.shape), -1)
            for idx in max_idx:
                feature_points.add((int(j * 32 + idx[1]), int(i * 32 + idx[0])))
                
            block_DoG = DoG[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
            max_idx = np.stack(np.unravel_index(np.argsort(np.ravel(-block_DoG))[:4], block_DoG.shape), -1)
            for idx in max_idx:
                feature_points.add((int(j * 32 + idx[1]), int(i * 32 + idx[0])))
            
    return feature_points  # in xy format
                
                
def get_harris_dog(gray):
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, :-1] = gray[:, 1:] - gray[:, :-1]
    grad_y[:-1, :] = gray[1:] - gray[:-1]
    # gradient_compute(((width + 31) // 32, (height + 31) // 32), (32, 32), (gray, grad_x, np.int(width), np.int(height), np.int(0)))
    # gradient_compute(((width + 31) // 32, (height + 31) // 32), (32, 32), (gray, grad_y, np.int(width), np.int(height), np.int(1)))

    kernel = get_gaussian_kernel(1.)
    # gradxx = ndimage.correlate(grad_x * grad_x, kernel)
    # gradxy = ndimage.correlate(grad_x * grad_y, kernel)
    # gradyy = ndimage.correlate(grad_y * grad_y, kernel)
    gradxx = signal.correlate2d(grad_x * grad_x, kernel, mode='same')
    gradxy = signal.correlate2d(grad_x * grad_y, kernel, mode='same')
    gradyy = signal.correlate2d(grad_y * grad_y, kernel, mode='same')
    harris_response = gradxx * gradyy - 2 * gradxy - 0.06 * (gradxx + gradyy) ** 2
    harris_response -= harris_response.min()
    harris_response /= harris_response.max()

    kernel_sqrt2 = get_gaussian_kernel(np.sqrt(2))
    DoG = np.abs(signal.correlate2d(gray, kernel, mode='same') - signal.correlate2d(gray, kernel_sqrt2, mode='same'))
    DoG -= DoG.min()
    DoG /= DoG.max()
    return harris_response, DoG


class FeatureExtraction:
    def __init__(self, img_paths, camera_paths, intrinsic):
        super().__init__()
        self.feature_points = []
        self.camera_feature_points = []
        self.rgbs = []
        self.grays = []
        scale = 0.1
        self.intrinsic = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]) @ intrinsic
        self.intrinsic_inv = np.linalg.inv(self.intrinsic)
        self.extrinsics = []
        self.camera_centers = []
        self.cells = []
        cell_size = 2
        for i, path in tqdm(enumerate(img_paths)):
            img = cv2.imread(path)
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.).astype(np.float32)
            img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
            img = np.asarray(img, dtype=np.float32)
            
            cells = {}
            for a in range(img.shape[0]):
                for b in range(img.shape[1]):
                    if a % 2 == 0 and b % 2 == 0:
                        cells[(a, b)] = Cell(i, b, a)
                    else:
                        cells[(a, b)] = cells[(a // 2 * 2, b // 2 * 2)]
            
            extrinsic = np.asarray(read_camera_extrinsic(camera_paths[i]))
            self.extrinsics.append(extrinsic)
            self.camera_centers.append(-extrinsic[:3, :3].T @ extrinsic[:3, -1])

            self.rgbs.append(img)
            gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, gray)
            # gray_convert(((width + 31) // 32, (height + 31) // 32), (32, 32), (img, gray, np.int(width), np.int(height)))
            
            harris, DoG = get_harris_dog(gray)
            self.grays.append(gray)
            raw_points = get_feature_points(harris, DoG)
            feature_poionts = np.array([[pt[0], pt[1], 1] for pt in raw_points])
            camera_feature_points = (self.intrinsic_inv @ feature_poionts.T).T
            self.feature_points.append(feature_poionts)
            self.camera_feature_points.append(camera_feature_points)
            for idx, pt in enumerate(feature_poionts):
                cells[(pt[1], pt[0])].feature_points_idx.append(idx)
                
            self.cells.append(cells)
        
        self.Fs = {}
        for i in range(len(self.grays)):
            for j in range(len(self.grays)):
                if i == j:
                    continue
                self.Fs[(i, j)] = np.asarray(compute_F_from_P(self.extrinsics[i], self.extrinsics[j]))
                
        self.patches = []
                
    
    def get_zncc_grad_latter(self, patch, idx1, idx2):
        _, zncc_grad = calc_zncc_grad(self.grays[idx1], self.grays[idx2], 
                                patch.project_onto_img(self.intrinsic, self.extrinsics[idx1]), 
                                patch.project_onto_img(self.intrinsic, self.extrinsics[idx2]))  # N x 2

        dp2dp = patch.projection_grad(self.intrinsic, self.extrinsics[idx2])  # N x 2 x 3
        dpdk, dpdangle = patch.pts_param_grad()
        dp2dk = np.sum(dp2dp * dpdk[:, None], -1)
        dp2dangle = np.sum(dp2dp[..., None] * dpdangle[:, None], 2)
        
        dznccdk = np.sum(zncc_grad * dp2dk)
        dznccdangle = np.sum(np.sum(zncc_grad[..., None] * dp2dangle, 0), 0)
        return -np.concatenate([dznccdk[..., None], dznccdangle], -1)

    def gp_grad(self, patch, i, v_star):
        res = np.mean(np.stack([self.get_zncc_grad_latter(patch, i, k) for k in v_star if k != i]), 0)
        return res

    def gp(self, patch, i, v_star):
        res = np.mean(np.stack([1. - calc_zncc(self.grays[i], self.grays[k], patch.project_onto_img(self.intrinsic, self.extrinsics[i]), patch.project_onto_img(self.intrinsic, self.extrinsics[k])) for k in v_star if k != i]))
        return res
    
    
    def matching(self):
        feature_points_mask = [np.ones((self.feature_points[i].shape[0]), dtype=np.bool) for i in range(len(self.grays))]
        for i in range(len(self.grays) - 1):
            for camera_point in tqdm(self.camera_feature_points[i][feature_points_mask[i]]):
                corrs = []
                for j in range(len(self.grays)):
                    if i == j:
                        continue
                    epiline = self.intrinsic_inv.T @ self.Fs[(i, j)] @ camera_point
                    epiline /= np.linalg.norm(epiline[:2])
                    dist2epiline = np.abs(np.sum(self.feature_points[j][feature_points_mask[j]] * epiline[None], -1))
                    close_pts = self.camera_feature_points[j][feature_points_mask[j]][dist2epiline < 3]
                    
                    for pt in close_pts:
                        c = np.asarray(opt_triangulate([self.extrinsics[i], self.extrinsics[j]], [[camera_point[0], camera_point[1], pt[0], pt[1]]])[0])
                        c = c[:3]
                        corrs.append((j, pt, c, np.linalg.norm(c - self.camera_centers[i])))
                

                    # if len(close_pts) > 10:
                    #     img = self.rgbs[j].copy()
                    #     img_ref = self.rgbs[i].copy()
                    #     cv2.circle(img_ref, (self.feature_points[i][k][0], self.feature_points[i][k][1]), 3, color=(255, 0, 0), thickness=-1)
                    #     for point in close_pts:
                    #         img = cv2.circle(img, (int(point[0]), int(point[1])), 1, color=(255, 0, 0), thickness=-1)
                        
                    #     cv2.imshow('img', img[:, :, ::-1])
                    #     cv2.imshow('img_ref', img_ref[:, :, ::-1])
                    #     cv2.waitKey()
                        
                corrs.sort(key=lambda x: x[3])
                for (j, pt, c, dist) in corrs:
                    patch = Patch(c, self.intrinsic_inv, self.extrinsics[i], i)
                    
                    # first round
                    proj_i = patch.project_onto_img(self.intrinsic, self.extrinsics[i])

                    c = patch.cam_center + patch.k * patch.d
                    co = c[None] - np.stack(self.camera_centers)
                    co /= np.linalg.norm(co, axis=-1, keepdims=True)
                    cos = co @ sph2norm(patch.theta, patch.phi)
                    vp = list(np.where(cos > 0.5)[0])
                    
                    v_star = []
                    for cand in vp:
                        proj_cand = patch.project_onto_img(self.intrinsic, self.extrinsics[cand])
                        zncc = calc_zncc(self.grays[i], self.grays[cand], proj_i, proj_cand)
                        
                        if 1. - zncc < 0.6:
                            v_star.append(cand)
                    
                    if len(v_star) < 2:
                        continue
                    
                    def gp(x):
                        backup = patch.k, patch.theta, patch.phi
                        patch.k, patch.theta, patch.phi = x
                        res = self.gp(patch, patch.ref_idx, v_star)
                        patch.k, patch.theta, patch.phi = backup
                        return res
                    
                    def gp_grad(x):
                        backup = patch.k, patch.theta, patch.phi
                        patch.k, patch.theta, patch.phi = x
                        res = self.gp_grad(patch, patch.ref_idx, v_star)
                        patch.k, patch.theta, patch.phi = backup
                        return res
                    
                    # print(patch.k, patch.theta, patch.phi)
                    best_x = optimize.fmin_cg(gp, np.array([patch.k, patch.theta, patch.phi]), fprime=gp_grad, disp=False)
                    patch.k, patch.theta, patch.phi = best_x
                    # print(patch.k, patch.theta, patch.phi)
                    
                    # conjugate gradient descent my implementation
                    # last_grad = None
                    # for it in range(100): 
                    #     if it == 0:
                    #         last_grad = gp_grad()
                    #         p = -last_grad
                    #         last_p = p
                    #     else:
                    #         beta = np.dot(last_grad, last_grad) / np.dot(last_last_grad, last_last_grad)
                    #         p = -last_grad + beta * last_p
                            
                    #     gp_before = gp()
                    #     grad_before = gp_grad()
                        
                    #     alpha = 1.
                    #     k, theta, phi = patch.k, patch.theta, patch.phi
                    #     patch.k = float(k + alpha * p[0])
                    #     patch.theta = float(theta + alpha * p[1])
                    #     patch.phi = float(phi + alpha * p[2])
                    #     patch.refresh()
                    #     while gp() > gp_before + 1 / 2 * alpha * np.dot(p, grad_before):
                    #         alpha *= 0.5
                    #         patch.k = float(k + alpha * p[0])
                    #         patch.theta = float(theta + alpha * p[1])
                    #         patch.phi = float(phi + alpha * p[2])
                    #         patch.refresh()
                        
                    #     last_p = p
                    #     last_last_grad = last_grad
                    #     last_grad = gp_grad()
                    #     if np.any(np.isnan(last_grad)):
                    #         import pdb; pdb.set_trace()
                    #     if np.linalg.norm(last_grad) < 0.1:
                    #         break
                    
                    
                        
                    # second round
                    proj_i = patch.project_onto_img(self.intrinsic, self.extrinsics[i])
                    
                    c = patch.cam_center + patch.k * patch.d
                    co = c[None] - np.stack(self.camera_centers)
                    co /= np.linalg.norm(co, axis=-1, keepdims=True)
                    cos = co @ sph2norm(patch.theta, patch.phi)
                    vp = list(np.where(cos > 0.5)[0])
                    
                    v_star = []
                    for cand in vp:
                        proj_cand = patch.project_onto_img(self.intrinsic, self.extrinsics[cand])
                        zncc = calc_zncc(self.grays[i], self.grays[cand], proj_i, proj_cand)
                        if 1. - zncc < 0.3:
                            v_star.append(cand)
                    
                    if len(v_star) < 2:
                        continue
                    
                    patch.v = vp
                    patch.v_star = v_star
                    
                    to_remove = []
                    for idx in vp:
                        proj = self.intrinsic @ self.extrinsics[idx] @ np.array([*c, 1]) 
                        proj = (proj[:2] / proj[-1]).astype(np.int)
                        if (proj[1], proj[0]) in self.cells[idx]:
                            self.cells[idx][(proj[1], proj[0])].q.append(patch)
                        
                    for idx in v_star:
                        proj = self.intrinsic @ self.extrinsics[idx] @ np.array([*c, 1]) 
                        proj = (proj[:2] / proj[-1]).astype(np.int)
                        if (proj[1], proj[0]) in self.cells[idx]:
                            self.cells[idx][(proj[1], proj[0])].q_star.append(patch)
                            patch.cells.append(self.cells[idx][(proj[1], proj[0])])
                            for pt_idx in self.cells[idx][(proj[1], proj[0])].feature_points_idx:
                                to_remove.append((idx, pt_idx))
                        
                    for (idx, pt_idx) in to_remove:
                        feature_points_mask[idx][pt_idx] = False
                        
                    self.patches.append(patch)
        
        colors = np.zeros((len(self.patches), 3))
        vis = self.rgbs[0].copy()
        for i, patch in enumerate(self.patches):
            pts = patch.project_onto_img(self.intrinsic, self.extrinsics[0])
            pt = pts[pts.shape[0] // 2].astype(np.int)
            if pt[0] >= 0 and pt[1] >= 0 and pt[0] < self.grays[0].shape[1] and pt[1] < self.grays[0].shape[0]:
                vis = cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, color=(255, 0, 0), thickness=-1)
                colors[i] = self.rgbs[0][pt[1], pt[0]]
        cv2.imshow('patch', cv2.resize(vis[..., ::-1], (0, 0), fx=4, fy=4))
        draw_patches(self.patches, colors)
        # cv2.waitKey()
        
    def is_neighbor(self, patch1, patch2):
        center1 = patch1.cam_center + patch1.k * patch1.d
        depth1 = (self.extrinsics[patch1.ref_idx] @ np.array([*center1, 1]))[-1]
        
        center2 = patch2.cam_center + patch2.k * patch2.d
        depth2 = (self.extrinsics[patch2.ref_idx] @ np.array([*center2, 1]))[-1]
        
        rho1 = 2 * self.intrinsic_inv[0, 0] * (depth1 + depth2) / 2
        return np.abs(np.dot(center1 - center2, sph2norm(patch1.theta, patch1.phi))) + np.abs(np.dot(center1 - center2, sph2norm(patch2.theta, patch2.phi))) < 2 * rho1

    def expand(self):
        it = 1
        queue = self.patches.copy()
        while len(queue) > 0:
            p = queue.pop(0)
            for cell in p.cells:
                C = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        idx = (cell.y + i * 2, cell.x + j * 2)
                        if idx in self.cells[cell.ref_idx]:
                            C.append(self.cells[cell.ref_idx][idx])
                
                # cp = p.cam_center + p.k * p.d
                # p_depth = (self.extrinsics[p.ref_idx] @ np.array([*cp, 1]))[-1]
                C_filtered = []
                for c in C:
                    neccessary = True
                    if len(c.q) > 0:
                        for pp in c.q:
                            # cpp = pp.cam_center + pp.k * pp.d
                            # pp_depth = (self.extrinsics[pp.ref_idx] @ np.array([*cpp, 1]))[-1]
                            # rho1 = 2 * self.intrinsic_inv[0, 0] * (p_depth + pp_depth) / 2
                            # if np.abs(np.dot(cp - cpp, sph2norm(p.theta, p.phi))) + np.abs(np.dot(cp - cpp, sph2norm(pp.theta, pp.phi))) < 2 * rho1:
                            if self.is_neighbor(p, pp):
                                neccessary = False
                                break
                        
                    if neccessary and len(c.q_star) > 0:
                        for pp in c.q_star:
                            if self.gp(pp, pp.ref_idx, pp.v_star) < 0.3:
                                neccessary = False
                                break
                            
                    if neccessary:
                        C_filtered.append(c)
                
                C = C_filtered
                # print(len(C))
                for c in C:
                    pprime = Patch(patch=p)
                    
                    # set new pc
                    n = sph2norm(p.theta, p.phi)
                    p_center = p.cam_center + p.k * p.d
                    plane = -np.dot(n, p_center)
                    pt = np.array([c.x + 1, c.y + 1, 1])
                    pt_world = np.linalg.pinv(self.extrinsics[pprime.ref_idx]) @ self.intrinsic_inv @ pt
                    pt_world = pt_world[:3] / pt_world[-1]
                    cam2pt = pt_world - p.cam_center
                    pprime_center = -(plane + np.dot(n, p.cam_center)) / np.dot(n, cam2pt) * cam2pt + p.cam_center
                    # print(p_center)
                    # print(pprime_center)
                    pprime.d = pprime_center - p.cam_center
                    pprime.k = np.linalg.norm(pprime.d)
                    pprime.d /= pprime.k
                    
                    # import pdb; pdb.set_trace()
                    
                    proj_ref = pprime.project_onto_img(self.intrinsic, self.extrinsics[pprime.ref_idx])
                    v_star = []
                    for cand in pprime.v:
                        proj_cand = pprime.project_onto_img(self.intrinsic, self.extrinsics[cand])
                        zncc = calc_zncc(self.grays[pprime.ref_idx], self.grays[cand], proj_ref, proj_cand)
                        if 1. - zncc < 0.6:
                            v_star.append(cand)
                    pprime.v_star = v_star
                    if len(pprime.v_star) < 2:
                        continue
                    
                    def gp(x):
                        backup = pprime.k, pprime.theta, pprime.phi
                        pprime.k, pprime.theta, pprime.phi = x
                        res = self.gp(pprime, pprime.ref_idx, pprime.v_star)
                        pprime.k, pprime.theta, pprime.phi = backup
                        return res
                    
                    def gp_grad(x):
                        backup = pprime.k, pprime.theta, pprime.phi
                        pprime.k, pprime.theta, pprime.phi = x
                        res = self.gp_grad(pprime, pprime.ref_idx, pprime.v_star)
                        pprime.k, pprime.theta, pprime.phi = backup
                        return res
                    
                    # print(patch.k, patch.theta, patch.phi)
                    best_x = optimize.fmin_cg(gp, np.array([pprime.k, pprime.theta, pprime.phi]), fprime=gp_grad, disp=False)
                    pprime.k, pprime.theta, pprime.phi = best_x
                    
                    # determine depth test threshold
                    n = sph2norm(pprime.theta, pprime.phi)
                    pprime_center = pprime.cam_center + pprime.k * pprime.d
                    plane = -np.dot(n, pprime_center)
                    pt = np.array([c.x + 3, c.y + 1, 1])  # offset two pixels
                    pt_world = np.linalg.pinv(self.extrinsics[pprime.ref_idx]) @ self.intrinsic_inv @ pt
                    pt_world = pt_world[:3] / pt_world[-1]
                    cam2pt = pt_world - pprime.cam_center
                    delta = np.linalg.norm(-(plane + np.dot(n, pprime.cam_center)) / np.dot(n, cam2pt) * cam2pt + pprime.cam_center - pprime_center)
                    pprime_depth = (self.extrinsics[pprime.ref_idx] @ np.array([*pprime_center, 1]))[-1]
                    
                    for cand in range(len(self.grays)):
                        if cand == pprime.ref_idx:
                            continue
                        proj = self.intrinsic @ self.extrinsics[cand] @ np.array([*(pprime.cam_center + pprime.k * pprime.d), 1]) 
                        proj = (proj[:2] / proj[-1]).astype(np.int)
                        if (proj[1], proj[0]) in self.cells[cand]:
                            if len(self.cells[cand][(proj[1], proj[0])].q) > 0:
                                valid = True
                                for pp in self.cells[cand][(proj[1], proj[0])].q:
                                    pp = self.cells[cand][(proj[1], proj[0])].q[0]
                                    pp_proj = self.extrinsics[pp.ref_idx] @ np.array([*(pp.cam_center + pp.k * pp.d), 1])
                                    depth = pp_proj[-1]
                                    if pprime_depth > depth + delta:
                                        valid = False
                                        break
                                if valid:
                                    pprime.v.append(cand)
                    
                    pprime.v = list(set(pprime.v))
                    v_star = []
                    for cand in pprime.v:
                        proj_cand = pprime.project_onto_img(self.intrinsic, self.extrinsics[cand])
                        zncc = calc_zncc(self.grays[pprime.ref_idx], self.grays[cand], proj_ref, proj_cand)
                        if 1. - zncc < 0.3:
                            v_star.append(cand)
                    pprime.v_star = list(set(v_star))
                    
                    if len(pprime.v_star) < 2:
                        continue
                    
                    queue.append(pprime)
                    self.patches.append(pprime)
                    for idx in pprime.v:
                        proj = self.intrinsic @ self.extrinsics[idx] @ np.array([*pprime_center, 1]) 
                        proj = (proj[:2] / proj[-1]).astype(np.int)
                        if (proj[1], proj[0]) in self.cells[idx]:
                            self.cells[idx][(proj[1], proj[0])].q.append(pprime)
                        
                    for idx in pprime.v_star:
                        proj = self.intrinsic @ self.extrinsics[idx] @ np.array([*pprime_center, 1]) 
                        proj = (proj[:2] / proj[-1]).astype(np.int)
                        if (proj[1], proj[0]) in self.cells[idx]:
                            self.cells[idx][(proj[1], proj[0])].q_star.append(pprime)
                            pprime.cells.append(self.cells[idx][(proj[1], proj[0])])
            
                    print('processed one: {}, current queue length: {}'.format(len(self.patches), len(queue)))
                    
                    colors = np.zeros((len(self.patches), 3))
                    vis = self.rgbs[0].copy()
                    for i, patch in enumerate(self.patches):
                        pts = patch.project_onto_img(self.intrinsic, self.extrinsics[0])
                        pt = pts[pts.shape[0] // 2].astype(np.int)
                        if pt[0] >= 0 and pt[1] >= 0 and pt[0] < self.grays[0].shape[1] and pt[1] < self.grays[0].shape[0]:
                            vis = cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, color=(255, 0, 0), thickness=-1)
                            colors[i] = self.rgbs[0][pt[1], pt[0]]
                    cv2.imshow('patch', cv2.resize(vis[..., ::-1], (0, 0), fx=4, fy=4))
                    cv2.waitKey(10)
                    
            # if len(self.patches) > it * 100:
            #     it += 1
            #     draw_patches(self.patches, colors)
        
    def filtering(self):
        # first filter
        outlier_mask = np.zeros((self.patches.shape[0],), dtype=np.bool)
        for i, p in enumerate(self.patches):
            for cell in p.cells:
                non_nbrs = [pp for pp in cell.q_star if (not self.is_neighbor(p, pp) and pp != p)]
                if len(non_nbrs) == 0:
                    continue
                if len(p.v_star) * (1. - self.gp(p, p.ref_idx, p.v_star)) < np.sum([1. - self.gp(pp, pp.ref_idx, pp.v_star) for pp in non_nbrs]):
                    outlier_mask[i] = True
                    break
                
        for i in range(len(self.patches)):
            if outlier_mask[i]:
                p = self.patches[i]
                for cell in p.cells:
                    if p in cell.q:
                        cell.q.remove(p)
                    if p in cell.q_star:
                        cell.q_star.remove(p)
        self.patches = [patch for (i, patch) in enumerate(self.patches) if not outlier_mask[i]]
            
        # second filter
        outlier_mask = np.zeros((self.patches.shape[0],), dtype=np.bool)
        for i, p in enumerate(self.patches):
            # depth test
            n = sph2norm(p.theta, p.phi)
            p_center = p.cam_center + p.k * p.d
            plane = -np.dot(n, p)
            proj_cam = self.extrinsics[p.ref_idx] @ np.array([*p_center, 1])
            pt = self.intrinsic @ proj_cam
            pt = pt[:2] / pt[:-1]
            pt[0] += 2 # offset two pixels
            pt_world = np.linalg.pinv(self.extrinsics[p.ref_idx]) @ self.intrinsic_inv @ pt
            pt_world = pt_world[:3] / pt_world[-1]
            cam2pt = pt_world - p.cam_center
            delta = np.linalg.norm(-(plane + np.dot(n, p.cam_center)) / np.dot(n, cam2pt) * cam2pt + p.cam_center - p_center)
            p_depth = proj_cam[-1]
            
            count = 0
            for cand in p.v_star:  # include itself
                proj = self.intrinsic @ self.extrinsics[cand] @ np.array([*(p.cam_center + p.k * p.d), 1]) 
                proj = (proj[:2] / proj[-1]).astype(np.int)
                if (proj[1], proj[0]) in self.cells[cand]:
                    if len(self.cells[cand][(proj[1], proj[0])].q) > 0:
                        valid = True
                        for pp in self.cells[cand][(proj[1], proj[0])].q:
                            pp = self.cells[cand][(proj[1], proj[0])].q[0]
                            pp_proj = self.extrinsics[pp.ref_idx] @ np.array([*(pp.cam_center + pp.k * pp.d), 1])
                            depth = pp_proj[-1]
                            if p_depth > depth + delta:
                                valid = False
                                break
                if valid:
                    count += 1
            if count < 2:
                outlier_mask[i] = True
        
        for i in range(len(self.patches)):
            if outlier_mask[i]:
                p = self.patches[i]
                for cell in p.cells:
                    if p in cell.q:
                        cell.q.remove(p)
                    if p in cell.q_star:
                        cell.q_star.remove(p)
        self.patches = [patch for (i, patch) in enumerate(self.patches) if not outlier_mask[i]]
        
        # third filter
        outlier_mask = np.zeros((self.patches.shape[0],), dtype=np.bool)
        for i, p in enumerate(self.patches):
            adjacents = [p]
            for cell in p.cells:
                if cell.ref_idx not in p.v:
                    continue
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        coord = (cell.y + n * 2, cell.x + m * 2)
                        if coord in self.cells[cell.ref_idx]:
                            adjacents.extend(self.cells[cell.ref_idx][coord].q_star)
            adjacents = list(set(adjacents))
            nbr_ratio = len([self.is_neighbor(p, adj) for adj in adjacents]) / len(adjacents)
            if nbr_ratio < 0.25:
                outlier_mask[i] = True
        
        for i in range(len(self.patches)):
            if outlier_mask[i]:
                p = self.patches[i]
                for cell in p.cells:
                    if p in cell.q:
                        cell.q.remove(p)
                    if p in cell.q_star:
                        cell.q_star.remove(p)
        self.patches = [patch for (i, patch) in enumerate(self.patches) if not outlier_mask[i]]
                
if __name__ == "__main__":
    

    fe = FeatureExtraction(glob('Benchmarking_Camera_Calibration_2008/fountain-P11/images/00*.jpg')[:2], 
                           glob('Benchmarking_Camera_Calibration_2008/fountain-P11/gt_dense_cameras/00*.jpg.camera')[:2], 
                           np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]]))
    
    fe.matching()
    fe.expand()
    # fe = FeatureExtraction(img)
    # gray = fe.gray.to_numpy()
    

    # cv2.imshow('gray', np.asnumpy(gray))
    # cv2.imshow('dog', np.asnumpy(DoG))
    # cv2.imshow('harris', np.asnumpy(harris_response))
    # cv2.waitKey()
    
    
    
    # print(len(feature_points))
    img = fe.rgbs[0]
    for point in fe.feature_points[0]:
        img = cv2.circle(img, (point[0], point[1]), 1, color=(0, 0, 255), thickness=-1)
    
    cv2.imshow('img', img[:, :, ::-1])
    cv2.waitKey()