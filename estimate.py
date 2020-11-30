# Usage: -> % python -B estimate.py Data/five_position_classes_head/r45/valid/6.pcd

import open3d as o3d
from model import PointNet
from dataset import Normalize
import torch
import numpy as np
import sys

# def estimate(path, model):
#     # TODO: 推測メソッドモジュールの実装


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print('loading model...')
    pointnet = PointNet(classes=5)
    pointnet.load_state_dict(torch.load('nbs/pnt_model_500.pth', map_location=device))
    pointnet.to(device)
    pointnet.eval()

    path = sys.argv[1]
    print(f"loading {path}...")
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points)

    points = Normalize()(points)
    X = torch.from_numpy(points)
    X = X.unsqueeze(0)
    X_tensor = X.to(device).float()
    # print(X_tensor.size())
    # print(X_tensor.device)

    with torch.no_grad():
        y, __, __ = pointnet(X_tensor.transpose(1,2))
        _, pred = torch.max(y.data, 1)

    label = {0: '0', 1: 'l45', 2: 'l90', 3: 'r45', 4: 'r90'}
    print(f"predicted: {label[int(pred)]}")