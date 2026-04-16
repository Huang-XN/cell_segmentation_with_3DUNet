import os
#import numpy as np
#import pandas as pd
#from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic
import torch

#import torch.nn.functional as F

from datasets import EarlyEmbryoDataset

path = "/home/huangxn/Desktop/Playground/virtual_embryo/heem/preimp_mouse/early_mouse/dataset/normalized_data"

dataset = EarlyEmbryoDataset(root=path)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    num_workers=4
)

# img_path = os.path.join(path,"train",folders[0],"images",folders[0]+"_image_0001.npy")
# folders = sorted(os.listdir(train_path))
'''shapes_data = []

for idx, name in enumerate(folders):
    img_path = os.path.join(train_path, name, "images", f"{name}_image_0001.npy")
    try:
        with open(img_path, 'rb') as f:
            version = read_magic(f)
            if version[0] == 1:
                shape, _, depth = read_array_header_1_0(f)
            elif version[0] == 2:
                shape, _, depth = read_array_header_2_0(f)
            else:
                # Fallback to full load for very old or unusual .npy files
                shape = np.load(img_path).shape
                depth = np.load(img_path).dtype
    except Exception as e:
        print(f"Warning: Could not read {img_path} - {e}")
        continue  # Skip problematic files
    
    z, y, x = shape
    shapes_data.append({
        'Sample_Index': name,
        'Dimension_Z': int(z),
        'Dimension_Y': int(x),
        'Dimension_X': int(y),
        'Total_Voxels': int(z * x * y),
        'Bit_Depth': depth,
        'Min':,
        'Max':
    })

    if idx % 100 == 0:
        print(f"Processed {idx} samples...")
    

df = pd.DataFrame(shapes_data)
csv_filename = "3d_volume_shapes_v2.csv"
df.to_csv(csv_filename, index=False)'''
'''print(folders[:2])
print(len(folders))
print(img_path)

img = np.load(img_path)
print(img.shape)

img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float16)


resized_img = F.interpolate(img_tensor, 
                            size = (128, 256, 256),
                            mode = 'trilinear')

print(resized_img.shape)

resized_img_np = resized_img.numpy()

np.save("resized_img.npy",resized_img)'''


