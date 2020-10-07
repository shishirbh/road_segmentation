import cv2
from pathlib import Path

inp = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop\Images")

op = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\COMB_CROP_RESIZED\Images")

for x in inp.iterdir():
    img = cv2.imread(str(x))
    resized = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)
    # gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    op_path = str(op) + "\\" + x.name
    cv2.imwrite(str(op_path), resized)