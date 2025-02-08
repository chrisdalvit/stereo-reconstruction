import argparse
import open3d as o3d

#
# Render a point cloud from a .ply file
#

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)

args = parser.parse_args()
pcd = o3d.io.read_point_cloud(args.file)
o3d.visualization.draw_geometries([pcd])