import argparse
import open3d
import os
import shutil

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Scene name")
parser.add_argument("-p", "--path", help="PLY source path")
parser.add_argument("-d", "--destination", help="Destination root")
args = parser.parse_args()

# Fix point cloud rgb values
ply = f"{args.path}"
with open(ply, 'r') as file:
    lines = file.readlines()

corrected_lines = []
header_passed = False
for line in lines:
    if header_passed:
        parts = line.split()
        if len(parts) == 6:
            x, y, z, r, g, b = parts
            r, g, b = (
                min(max(0, int(r)), 255),
                min(max(0, int(g)), 255),
                min(max(0, int(b)), 255),
            )
            corrected_lines.append(f"{x} {y} {z} {r} {g} {b}\n")
        else:
            corrected_lines.append(line)
    else:
        corrected_lines.append(line)
        if line.strip() == "end_header":
            header_passed = True  # Indicates that the next lines are vertex data

with open(ply, 'w') as file:
    file.writelines(corrected_lines)

# Read point cloud from source path
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

    for line in lines[1:]:
        parts = line.strip().split()
        modified_line = " ".join(parts) + " 0.000000 0.000000 0.000000\n"
        f.write(modified_line)
    f.truncate()

shutil.copyfile(f"{root}/{args.name}.txt", f"{root}/Annotations/{args.name}.txt")
with open(f"{root}_alignmentAngle.txt", "w") as f:
    f.write(f"{args.name} 0")
