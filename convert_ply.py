import argparse
import open3d
import os
import shutil

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-n", "--name", help="Scene name")
parser.add_argument("-p", "--path", help="PLY source path")
parser.add_argument("-d", "--destination", help="Destination root")

# Read arguments from command line
args = parser.parse_args()

# Read point cloud from source path
ply = f"{args.path}"
pcd = open3d.io.read_point_cloud(ply)

# Restructure to s3dis dataset format
root = f"{args.destination}/{args.name}/{args.name}/{args.name}"
os.makedirs(f"{root}/Annotations", exist_ok=True)

open3d.io.write_point_cloud(
    pointcloud=pcd,
    filename=f"{root}/{args.name}.pts",
    format="xyz",
)

os.rename(f"{root}/{args.name}.pts", f"{root}/{args.name}.txt")
with open(f"{root}/{args.name}.txt", "r+") as f:
    lines = f.readlines()
    f.seek(0)

    for line in lines:
        parts = line.strip().split()
        modified_line = " ".join(parts) + " 0.000000 0.000000 0.000000\n"
        f.write(modified_line)
    f.truncate()

shutil.copyfile(f"{root}/{args.name}.txt", f"{root}/Annotations/{args.name}.txt")
with open(f"{root}_alignmentAngle.txt", "a") as f:
    f.write(f"{args.name} 0")
