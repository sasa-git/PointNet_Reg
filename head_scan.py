# five_position_class
# 顔のデータを撮影して指定のクラスにpcdファイルを保存
# Usage: -> % python3 get_face_scan.py TestData/five_position_classes/r45/valid
# 第二引数で保存名を指定
# Usage: -> % python3 get_face_scan.py TestData/five_position_classes/r45/valid test

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objects as go
import plotly.express as px
import plotly
from path import Path
import os
import sys
import time
from argparse import ArgumentParser

def parser():
    usage = f'Usage: python {__file__} OUTPUT_DIR FILE_NAME [--verbose] [--time_watch]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('class_dir', type=str, help='output dir')
    argparser.add_argument('-f', '--fname', type=str, help='output file name')
    argparser.add_argument('-a', '--auto', action='store_true', help='enable auto crop while DBSCAN ラベルは各撮影環境にて設定')
    argparser.add_argument('-l', '--label', type=int, default=0, help='set label when auto mode. default: 0 auto mode使用時に設定')
    argparser.add_argument('-v', '--verbose', action='store_true', help='enable transform visualize mode')
    argparser.add_argument('-t', '--time_watch', action='store_true', help='enable prosess time')
    args = argparser.parse_args()
    return args

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def show_img(img, dep_img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    for axe, image in zip(axes, [img, dep_img]):
        axe.imshow(image)
    plt.show()
    plt.close()

def dbscan(pcd, auto_mode=False, label_val=0):
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=80, print_progress=True))
    print(f"pcd.points.shape(): {np.array(pcd.points).shape}")
    print(f"labels.shape(): {labels.shape}")

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    print("First 5 labels:",labels[:10])

    if auto_mode ==False:
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])

    for label in np.unique(labels):
        mask = labels == label
        
        part_of_pcd = o3d.geometry.PointCloud()
        part_of_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
        if auto_mode == False:
            print(f"label: {label}")
            o3d.visualization.draw_geometries([part_of_pcd])
        points_size = len(np.array(part_of_pcd.points))
        if label >= 0 and label <= 3 and points_size > 1000:
            val = label
        if label >= 10:
            break
    
    if auto_mode == True:
        val = label_val
    else:
        print("choose face label:")
        val = int(input())

    mask = labels == val
    face = o3d.geometry.PointCloud()
    face.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
    print(f"you choosed: {val}")
    # o3d.visualization.draw_geometries([face])
    return face

def crop(points, pcd):
    # show by plotly
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    pcshow(x, y, z)

    crop_point = float(input("Crop point: "))

    # new_points = points[points[:, 1] > -0.12][:600]
    # new_points = points[points[:, 1] > -0.06][:600] # Realsenseから60cm
    new_points = points[points[:, 1] > crop_point][:600] # 顔だけの場合は大体600ポイントぐらい
    pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd

def crop_bust(points, pcd, limit):
    assert len(points) > limit, f"expected len(points) > {limit}, but was {len(points)}:("
    sort = points[:, 1].argsort()
    new_points = points[sort][::-1][:limit]
    pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd

args = parser()

# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

# get camera intrinsics
# http://www.open3d.org/docs/0.9.0/python_api/open3d.geometry.PointCloud.html?highlight=create#open3d.geometry.PointCloud.create_from_depth_image
# ※PrimeSenseDefaultを指定していますが、本来ここはRealSense D435iでカメラキャリブレーションを行い、内部パラメータを求めた上で、内部パラメータ(instric)の行列を与えるべき
# ->https://qiita.com/tishihara/items/f14c6b5db98d44f4d4ae
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

print('show camera preview...')
while True:
    # フレーム待ち(Color & Depth)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not depth_frame or not color_frame:
        continue

    img = np.array(color_frame.get_data())
    dep_img = np.array(depth_frame.get_data())

    cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('color image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) != -1:
        print('shot!')
        cv2.destroyAllWindows()
        break

# args = parser()
# time_watch = False
time_watch = args.time_watch
show_verbose = args.verbose

if time_watch == True:
    print('Watching Time')
    start_time = time.time()

# Depthイメージだけで点群生成
print(dep_img.shape)
color_image = o3d.geometry.Image(img)
depth_image = o3d.geometry.Image(dep_img)

# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)

pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, pinhole_camera_intrinsic)
# 回転
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, -0.5])

if show_verbose == True:
    print('your taked pcd:')
    o3d.visualization.draw_geometries([pcd, mesh_frame])

face_pcd = dbscan(pcd, auto_mode=args.auto, label_val=args.label)

# Downsample
down_sample_face = face_pcd.voxel_down_sample(voxel_size=0.01)

# Limited points
points = np.array(down_sample_face.points)
print(f"points.shape: {points.shape}")

if show_verbose == True:
    print('Down sampled face...')
    o3d.visualization.draw_geometries([down_sample_face])

# 顔だけ切り取る場合はコメントを外す
# down_sample_face = crop(points, down_sample_face)
down_sample_face = crop_bust(points, down_sample_face, 1000)

print('Limited points face...')
shape = np.array(down_sample_face.points).shape
print(f'down_sample_face.shape: {shape}')
if not time_watch == True:
    o3d.visualization.draw_geometries([down_sample_face])

if time_watch == True:
    exec_time = time.time() - start_time
    print(f'exec time: {exec_time * 1000:.3f}ms"')

# root_path = Path("TestData/five_faces_class")
# root_path = Path("TestData/five_position_classes")
# root_path = Path(sys.argv[1]) # ←"TestData/five_position_classes/0/train"
root_path = Path(args.class_dir)

# if len(sys.argv) == 3:
#     name = sys.argv[2] + ".pcd"
# else:
#     name = str(len(os.listdir(root_path))) + ".pcd"

if args.fname:
    name = args.fname + ".pcd"
else:
    name = str(len(os.listdir(root_path))) + ".pcd"

path = root_path/name
# path = input('Put path and filename(ex...[r45/train/x]):')

# face_pcd 首下もある
# down_sample_face 首上だけ
o3d.io.write_point_cloud(path, down_sample_face)
print(f'saved at [{path}]')

# ストリーミング停止
pipeline.stop()

### MEMO
# 今後は首上と胸上のデータ両方保存してみる?
# 胸上は大体1800~2000ぐらい→y軸についてソートして1700まで点を削減してみるのがいいかも