import open3d as o3d
import numpy as np


def draw_cameras(Ps, size=0.1):
    boxs = []
    for P in Ps:
        R = P[:3, :3]
        t = P[:3, -1]
        
        pos = np.eye(4)
        pos[:3, :3] = R.T
        pos[:3, -1] = -R.T @ t
        mesh_box = o3d.geometry.TriangleMesh.create_box(width=size * 2,
                                                    height=size,
                                                    depth=size)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        mesh_box.transform(pos)
        
        boxs.append(mesh_box)
        
    o3d.visualization.draw_geometries(boxs)
    
    
    