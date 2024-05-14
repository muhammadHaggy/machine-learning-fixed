import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-n", "--name", help = "NPY Prediction Path")
 
# Read arguments from command line
args = parser.parse_args()

label_path = f"npy/{args.name}.npy"
full_ply = f"3dgs_full_ply/{args.name}.ply"
project_name = args.name

pred_labels = np.load(label_path)
plydata = PlyData.read(full_ply)

segmented_items = {

}
properties = []


for property in plydata.elements[0].properties:
    properties.append((property.name, 'f4'))

counter = 0

for idx, item in enumerate(pred_labels):
    print(f"{counter}/{len(pred_labels)}")
    if segmented_items.get(str(item)) == None:
        segmented_items[str(item)] = [plydata.elements[0].data[idx]]
    else:
        segmented_items[str(item)] += [plydata.elements[0].data[idx]]
    counter+=1

print(segmented_items.keys())

for key in segmented_items.keys():
    points = segmented_items[key]
    vertex = np.array(points
                    ,dtype=properties)

    plydata = PlyElement.describe(vertex, 'vertex')
    dir_name = f"segment_result_ply/{project_name}"
    os.makedirs(dir_name, exist_ok=True)
    PlyData([plydata]).write(f'{dir_name}/{project_name}_{key}.ply')
