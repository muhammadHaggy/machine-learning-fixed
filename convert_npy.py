import numpy as np
import open3d
import argparse
import os


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--gaussian", help="Gaussian Splatting PLY source path")
parser.add_argument("--scene", help="PTv3 NPY source path")
parser.add_argument("--destination", help="Conversion PLY destination path")
parser.add_argument("--name", help="NPY Prediction Path")
args = parser.parse_args()

gaussian_ply = f"{args.gaussian}"
scene_npy = f"{args.scene}"

project_name = f"{args.name}"
dir_name = f"{args.destination}"
os.makedirs(dir_name, exist_ok=True)

# Handle class label colors
_label_to_color_uint8 = {
    0: [158, 218, 228],
    1: [151, 223, 137],
    2: [174, 198, 232],
    3: [255, 187, 128],
    4: [254, 127, 13],
    5: [196, 176, 213],
    6: [213, 39, 40],
    7: [188, 189, 35],
    8: [255, 152, 151],
    9: [140, 86, 74],
    10: [196, 156, 147],
    11: [148, 103, 188],
    12: [0, 0, 0],
}

_label_to_color = dict(
    [
        (label, (np.array(color_uint8).astype(np.float64) / 255.0).tolist())
        for label, color_uint8 in _label_to_color_uint8.items()
    ]
)

# Load source data
pred_labels = np.load(scene_npy)
pred_labels_colors = np.array([_label_to_color[l] for l in pred_labels])
plydata = open3d.io.read_point_cloud(gaussian_ply)

# Generate final point cloud
pcd = open3d.geometry.PointCloud()
pcd.points = plydata.points
pcd.colors = open3d.utility.Vector3dVector(pred_labels_colors)

open3d.io.write_point_cloud(f"{dir_name}/{project_name}.ply", pcd)
