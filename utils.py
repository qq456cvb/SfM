import numpy as np
import open3d as o3d


def sample_pixels(img, pts):
    lx = np.floor(pts[:, 0]).astype(np.int)
    ly = np.floor(pts[:, 1]).astype(np.int)
    hx = lx + 1
    hy = ly + 1
    wx = pts[:, 0] - lx
    wy = pts[:, 1] - ly
    valid_mask = np.logical_and(np.logical_and(lx >= 0, hx < img.shape[1]), np.logical_and(ly >= 0, hy < img.shape[0]))
    pixels = np.zeros((pts.shape[0]))
    if np.sum(valid_mask) == 0:
        return pixels
    pixels[valid_mask] = img[ly[valid_mask], lx[valid_mask]] * (1 - wx[valid_mask]) * (1 - wy[valid_mask]) \
        + img[hy[valid_mask], lx[valid_mask]] * (1 - wx[valid_mask]) * wy[valid_mask] \
        + img[ly[valid_mask], hx[valid_mask]] * wx[valid_mask] * (1 - wy[valid_mask]) \
        + img[hy[valid_mask], hx[valid_mask]] * wx[valid_mask] * wy[valid_mask]
    return pixels

def sample_pixels_grad(img, pts):
    lx = np.floor(pts[:, 0]).astype(np.int)
    ly = np.floor(pts[:, 1]).astype(np.int)
    hx = lx + 1
    hy = ly + 1
    wx = pts[:, 0] - lx
    wy = pts[:, 1] - ly
    valid_mask = np.logical_and(np.logical_and(lx >= 0, hx < img.shape[1]), np.logical_and(ly >= 0, hy < img.shape[0]))
    grad_x = np.zeros((pts.shape[0],))
    grad_y = np.zeros((pts.shape[0],))
    if np.sum(valid_mask) == 0:
        return np.stack([grad_x, grad_y], -1)
    grad_x[valid_mask] = img[ly[valid_mask], lx[valid_mask]] * (-1) * (1 - wy[valid_mask]) \
        + img[hy[valid_mask], lx[valid_mask]] * (-1) * wy[valid_mask] \
        + img[ly[valid_mask], hx[valid_mask]] * (1 - wy[valid_mask]) \
        + img[hy[valid_mask], hx[valid_mask]] * wy[valid_mask]
    grad_y[valid_mask] = img[ly[valid_mask], lx[valid_mask]] * (1 - wx[valid_mask]) * (-1) \
        + img[hy[valid_mask], lx[valid_mask]] * (1 - wx[valid_mask]) \
        + img[ly[valid_mask], hx[valid_mask]] * wx[valid_mask] * (-1) \
        + img[hy[valid_mask], hx[valid_mask]] * wx[valid_mask]
    return np.stack([grad_x, grad_y], -1)


def calc_zncc(img1, img2, pts1, pts2):
    pixels1 = sample_pixels(img1, pts1)
    pixels2 = sample_pixels(img2, pts2)
    return np.mean((pixels1 - np.mean(pixels1)) * (pixels2 - np.mean(pixels2))) / (np.std(pixels1) * np.std(pixels2) + 1e-3)
    
    
def calc_zncc_grad(img1, img2, pts1, pts2):
    pixels1 = sample_pixels(img1, pts1)
    grad_pts1 = sample_pixels_grad(img1, pts1)
    pixels2 = sample_pixels(img2, pts2)
    grad_pts2 = sample_pixels_grad(img2, pts2)
    std1 = np.std(pixels1)
    std2 = np.std(pixels2)
    norm_pixels1 = pixels1 - np.mean(pixels1)
    norm_pixels2 = pixels2 - np.mean(pixels2)
    n = pts1.shape[0]
    denom = std1 * std2 + 1e-3
    nom = np.mean(norm_pixels1 * norm_pixels2)
    grad1 = (1 / n * (-np.mean(norm_pixels2) + norm_pixels2) * denom \
        - nom * (-np.mean(norm_pixels1) + norm_pixels1) / (n * std1 + 1e-3) * std2) / denom ** 2
    grad2 = (1 / n * (-np.mean(norm_pixels1) + norm_pixels1) * denom \
        - nom * (-np.mean(norm_pixels2) + norm_pixels2) / (n * std2 + 1e-3) * std1) / denom ** 2
    return grad1[:, None] * grad_pts1, grad2[:, None] * grad_pts2


          
def draw_patches(patches, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.stack([patch.cam_center + patch.k * patch.d for patch in patches]))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud])
    return


def sph2norm(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
