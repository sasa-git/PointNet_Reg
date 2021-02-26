import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # 回転する
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # 法線計算
        pcd.estimate_normals()

        # 指定したvoxelサイズでダウンサンプリング
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

        distances = voxel_down_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        # メッシュ化
        radii = [radius, radius * 2]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(voxel_down_pcd, o3d.utility.DoubleVector(radii))

        # 再度法線計算
        mesh.compute_vertex_normals()

        # 一応データ保存
        o3d.io.write_triangle_mesh("output.ply", mesh)

        # メッシュデータの表示
        o3d.visualization.draw_geometries([mesh])

finally:
    # ストリーミング停止
    pipeline.stop()