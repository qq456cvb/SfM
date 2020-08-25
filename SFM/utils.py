import numpy as np
from SFM.visualization import draw_cameras

def read_camera_extrinsic(txt):
    lines = open(txt).readlines()
    extrinsic = lines[4:8]
    extrinsic = [[float(a) for a in line.split()] for line in extrinsic]
    extrinsic = np.array(extrinsic)
    rot = extrinsic[:3, :3]
    trans = extrinsic[-1]
    extrinsic = np.zeros((3, 4))
    extrinsic[:3, :3] = rot.T
    extrinsic[:3, -1] = -rot.T @ trans
    return extrinsic


if __name__ == "__main__":
    cameras = [read_camera_extrinsic('Benchmarking_Camera_Calibration_2008/castle-P30/gt_dense_cameras/00{:02d}.jpg.camera'.format(i)) for i in range(30)]
    draw_cameras(cameras, 0.5)