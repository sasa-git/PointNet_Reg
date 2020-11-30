import open3d as o3d
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from path import Path

from model import PointNet
from sklearn.metrics import confusion_matrix

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def __call__(self, data):
        return data[:self.output_size]

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        # x, y, z軸で平均を引く→各ベクトルの大きさの最大値で各要素を割る
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
        PointSampler(1000),
        Normalize(),
        ToTensor()
    ])

def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points)
    return points

# Custom Pytorch Datasetオブジェクトの作成

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms(), direct_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder:i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

        for category in self.classes.keys():
            new_dir = root_dir/category/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pcd'):
                    # PCDファイルからpcd.points読み込み
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)
    
    def __len__(self):
        return len(self.files)
    
    def __preproc__(self, path):
        points = read_pcd(path)
        if self.transforms:
            pointcloud = self.transforms(points)
        return pointcloud
    
    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        pointcloud = self.__preproc__(pcd_path)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}

if __name__ == '__main__':
    path = Path("Data/five_position_classes_head")
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print(classes)

    train_ds = PointCloudData(path)
    valid_ds = PointCloudData(path, valid=True, folder='valid')

    inv_classes = {i:cat for cat, i in train_ds.classes.items()}
    print(inv_classes)

    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[-1]['pointcloud'].size())
    print('first sample class: ', inv_classes[train_ds[0]['category']])

    print('sample size:', train_ds[-1]['pointcloud'].size())


    valid_loader = DataLoader(dataset=valid_ds, batch_size=5)
    pointnet = PointNet(classes=5)
    pointnet.load_state_dict(torch.load('nbs/pnt_model_500.pth'))
    pointnet.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
                    
            inputs, labels = data['pointcloud'].float(), data['category']
            outputs, __, __ = pointnet(inputs.transpose(1,2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.numpy())
            all_labels += list(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)