import numpy as np
from utils import sph2norm

class Cell:
    def __init__(self, ref_idx, x, y):
        super().__init__()
        self.q = []
        self.q_star = []
        self.ref_idx = ref_idx
        self.feature_points_idx = []
        self.x = x
        self.y = y
        

class Patch:
    def __init__(self, c=None, intrinsic_inv=None, extrinsic=None, ref_idx=None, patch=None):
        super().__init__()
        if patch is None:
            self.cam_center = -extrinsic[:3, :3].T @ extrinsic[:3, -1]
            n = c - self.cam_center # world coord
            self.k = np.linalg.norm(n)
            n /= self.k
            self.theta = np.arccos(n[-1])
            self.phi = np.arctan2(n[1], n[0])
            self.d = n.copy()
            self.r = ref_idx
            self.v = [ref_idx]
            self.v_star = []
            self.cells = []
            self.ref_idx = ref_idx
            
            # self.c = c  # world coord
            
            n_cam = extrinsic[:3, :3] @ c + extrinsic[:3, -1]
            z = n_cam[-1]
            # n_cam /= np.linalg.norm(n_cam)
            self.cross_x = extrinsic[0, :3]
            y_axis = np.cross(n, self.cross_x)
            x_axis = np.cross(y_axis, n)
            
            self.h = max(abs(intrinsic_inv[0, 0] * 5 * z / 4 / np.dot(extrinsic[0, :3], x_axis)), abs(intrinsic_inv[1, 1] * 5 * z / 4 / np.dot(extrinsic[1, :3], y_axis)))
            
            # # transform back to world coord
            # x_axis = extrinsic[:3, :3].T @ x_axis
            # y_axis = extrinsic[:3, :3].T @ y_axis
        else:
            self.cam_center = patch.cam_center.copy()
            self.k = patch.k
            self.theta = patch.theta
            self.phi = patch.phi
            self.d = patch.d
            self.r = patch.r
            self.v = patch.v.copy()
            self.v_star = patch.v_star.copy()
            self.cells = []
            self.ref_idx = patch.ref_idx
            self.cross_x = patch.cross_x
            self.h = patch.h
        
        
    def get_axis_from_n(self):
        n = sph2norm(self.theta, self.phi)
        y_axis = np.cross(n, self.cross_x)
        x_axis = np.cross(y_axis, n)
        return x_axis, y_axis
    
    # def refresh(self):
    #     self.c = self.cam_center + self.k * self.d
    #     self.n = np.array([np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta)])
    #     self.y_axis = -np.cross(self.cross_x, self.n)
    #     self.x_axis = np.cross(self.y_axis, self.n)
        
    def get_mesh_pts(self):
        c = self.cam_center + self.k * self.d
        x_axis, y_axis = self.get_axis_from_n()
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = c + self.h * i * x_axis + self.h * j * y_axis
                pts.append(pt)
        pts = np.stack(pts)
        return pts
        
    def project_onto_img(self, intrinsic, extrinsic):
        c = self.cam_center + self.k * self.d
        x_axis, y_axis = self.get_axis_from_n()
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = c + self.h * i * x_axis + self.h * j * y_axis
                pts.append(pt)
        pts = np.stack(pts)
        projection = pts @ extrinsic[:3, :3].T + extrinsic[:3, -1][None]
        projection = projection @ intrinsic.T
        projection = projection[:, :2] / projection[:, -1:]
        return projection
    
    def projection_grad(self, intrinsic, extrinsic):
        c = self.cam_center + self.k * self.d
        x_axis, y_axis = self.get_axis_from_n()
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                pt = c + self.h * i * x_axis + self.h * j * y_axis
                pts.append(pt)
        pts = np.stack(pts)
        projection = pts @ extrinsic[:3, :3].T + extrinsic[:3, -1][None]
        projection = projection @ intrinsic.T
        KR = intrinsic @ extrinsic[:3, :3]
        grad = np.stack([1 / projection[:, -1:] * KR[0][None] - projection[:, 0:1] / projection[:, -1:] ** 2 * KR[2][None],
                         1 / projection[:, -1:] * KR[1][None] - projection[:, 1:2] / projection[:, -1:] ** 2 * KR[2][None]], 1)  # N x 2 x 3
        return grad
    
    def pts_param_grad(self):
        n = np.array([np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta)])
        _, y_axis = self.get_axis_from_n()
        cross_x = self.cross_x
        theta = self.theta
        phi = self.phi
        dydn = -np.array([[0, -cross_x[2], cross_x[1]], [cross_x[2], 0, -cross_x[0]], [-cross_x[1], cross_x[0], 0]])
        dxdn = np.array([[0, -y_axis[2], y_axis[1]],
                               [y_axis[2], 0, -y_axis[0]],
                               [-y_axis[1], y_axis[0], 0]]) - np.array([[0, -n[2], n[1]],
                               [n[2], 0, -n[0]],
                               [-n[1], n[0], 0]]) @ dydn
        dndangle = np.array([[np.cos(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi)],
                             [np.cos(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)],
                             [-np.sin(theta), 0]])
        
        dpdns = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                dpdx = self.h * i
                dpdy = self.h * j
                dpdn = dpdx * dxdn + dpdy * dydn
                dpdns.append(dpdn)
        dpdns = np.stack(dpdns)  # N x 3 x 3
        dpdangles = np.einsum('nij,jk->nik', dpdns, dndangle)  # N x 3 x 2
        dpdk = np.ones((dpdangles.shape[0], 3)) * self.d[None]
        return dpdk, dpdangles
                