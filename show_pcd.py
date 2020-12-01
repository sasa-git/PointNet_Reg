# Usage: -> % python3 show_one_face.py TestData/five_position_classes/0/train/0.pcd 

import open3d as o3d
import numpy as np
import sys

path = sys.argv[1]
pcd = o3d.io.read_point_cloud(path)
points = np.array(pcd.points)

print(points.shape)
o3d.visualization.draw_geometries([pcd])