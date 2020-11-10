import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("Data/five_faces_class/0/0.pcd")
points = np.array(pcd.points)
print(points.shape)