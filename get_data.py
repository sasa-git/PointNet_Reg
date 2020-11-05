# five_position_class
# 顔のデータを撮影して指定のクラスにpcdファイルを保存

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objects as go
import plotly.express as px
import plotly
from path import Path

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

def dbscan(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))
    print(f"pcd.points.shape(): {np.array(pcd.points).shape}")
    print(f"labels.shape(): {labels.shape}")

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    print("First 5 labels:",labels[:10])
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    for label in np.unique(labels):
        mask = labels == label
        
        part_of_pcd = o3d.geometry.PointCloud()
        part_of_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
        print(f"label: {label}")
        o3d.visualization.draw_geometries([part_of_pcd])
        if label >= 3:
            break

    print("choose face label:")
    val = int(input())

    mask = labels == val
    face = o3d.geometry.PointCloud()
    face.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
    print(f"you choosed: {val}")
    o3d.visualization.draw_geometries([face])
    return face


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

print('your taked pcd:')
o3d.visualization.draw_geometries([pcd, mesh_frame])

face_pcd = dbscan(pcd)

# Downsample
# voxel_size=0.01---600pointsぐらい
# voxel_size=0.005---2400ぐらい
down_sample_face = face_pcd.voxel_down_sample(voxel_size=0.01)
print('Down sampled face...')
o3d.visualization.draw_geometries([down_sample_face])

# Limited points
points = np.array(down_sample_face.points)

# show by plotly
x, y, z = points[:, 0], points[:, 1], points[:, 2]
pcshow(x, y, z)

crop_point = float(input("Crop point: "))

# points = points[points[:, 1] > -0.12][:600]
# points = points[points[:, 1] > -0.06][:600] # Realsenseから60cm
points = points[points[:, 1] > crop_point][:600] # Realsenseから60cm
down_sample_face.points = o3d.utility.Vector3dVector(points)
print('Limited points face...')
o3d.visualization.draw_geometries([down_sample_face])


shape = np.array(down_sample_face.points).shape

print(f'face_pcd.shape: {shape}')

root_path = Path("TestData/five_faces_class")

path = input('Put path and filename:')
path += ".pcd"
o3d.io.write_point_cloud(root_path/path, face_pcd)
print(f'saved at [{root_path/path}].')

# ストリーミング停止
pipeline.stop()