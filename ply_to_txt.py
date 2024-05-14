import argparse
import open3d
import os

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--name", help = "Project name")
parser.add_argument("-r", "--root", help = "Project name")
 
# Read arguments from command line
args = parser.parse_args()

chair = f'{args.root}'

pcd = open3d.io.read_point_cloud(chair)
open3d.io.write_point_cloud(f"{args.name}.xyz", pcd, format="xyz")
os.rename(f"{args.name}.xyz", f"{args.name}.txt")
