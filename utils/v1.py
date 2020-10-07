from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.options.display.float_format = "{:,.4f}".format

# msk_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\output")
# pixel_list = []
# for x in msk_path.iterdir():

#     img = cv2.imread(str(x))
#     img = img.flatten()

#     unique, counts = np.unique(img, return_counts=True)
#     pixel_list.append([str(x.name), counts[1], (counts[1]/225000)])

# print(pixel_list)
# df = pd.DataFrame(pixel_list,columns =['filename', 'pixel_count', 'normalized_pixel_count']) 
# df.to_csv('info.csv')

df = pd.read_csv("C:\\Users\\nbhas\Desktop\Shishir\info.csv")
plt.hist(df['normalized_pixel_count']/10, bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9, 0.95,1])
plt.show()
 