import cv2
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path

dest_path = Path("C:\\Users\\nbhas\Desktop\Shishir\models-master\\research\deeplab\datasets\\road_map\SegmentationClass")
img_path = Path("C:\\Users\\nbhas\Desktop\Shishir\models-master\\research\deeplab\datasets\\road_map\seg")

# image = cv2.imread(img_path)
# histg = cv2.calcHist([image],[0],None,[256],[0,256])

# # find frequency of pixels in range 0-255 
# histr = cv2.calcHist([img],[0],None,[256],[0,256])   
# # show the plotting graph of an image 
# plt.plot(histr) 
# plt.show()
for x in img_path.iterdir():
    write_file = str(dest_path)+"\\"+ str(x.name)
    image = cv2.imread(str(x))
    image[np.where((image==[0,0,255]).all(axis=2))] = [128,128,192]
    cv2.imwrite(write_file, image)

