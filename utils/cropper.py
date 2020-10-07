import cv2
from pathlib import Path
import numpy as np
import shutil
import csv


def crop_copy_img(msk_path,img_path,img_dest_path,msk_dest_path):
    img = cv2.imread(str(img_path))
    msk = cv2.imread(str(msk_path),0)
    # msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    print(img.shape)

    quad1 = img[0:500, 0:500]
    msk_quad1 =  msk[0:500, 0:500]
    file_write_name_quad1 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-1.png")
    cv2.imwrite(file_write_name_quad1,quad1)
    msk_file_write_name_quad1 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-1.png")
    cv2.imwrite(msk_file_write_name_quad1, msk_quad1)

    quad2 = img[0:500, 500:1000]
    msk_quad2 = msk[0:500, 500:1000]
    file_write_name_quad2 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-2.png")
    cv2.imwrite(file_write_name_quad2,quad2)
    msk_file_write_name_quad2 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-2.png")
    cv2.imwrite(msk_file_write_name_quad2, msk_quad2)

    quad3 = img[0:500, 1000:1500]
    msk_quad3 = msk[0:500, 1000:1500]
    file_write_name_quad3 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-3.png")
    cv2.imwrite(file_write_name_quad3,quad3)
    msk_file_write_name_quad3 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-3.png")
    cv2.imwrite(msk_file_write_name_quad3, msk_quad3)

    quad4 = img[500:1000, 0:500]
    msk_quad4 = msk[500:1000, 0:500]
    file_write_name_quad4 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-4.png")
    cv2.imwrite(file_write_name_quad4,quad4)
    msk_file_write_name_quad4 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-4.png")
    cv2.imwrite(msk_file_write_name_quad4, msk_quad4)

    quad5 = img[500:1000, 500:1000]
    msk_quad5 = msk[500:1000, 500:1000]
    file_write_name_quad5 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-5.png")
    cv2.imwrite(file_write_name_quad5,quad5)
    msk_file_write_name_quad5 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-5.png")
    cv2.imwrite(msk_file_write_name_quad5, msk_quad5)

    quad6 = img[500:1000, 1000:1500]
    msk_quad6 = msk[500:1000, 1000:1500]
    file_write_name_quad6 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-6.png")
    cv2.imwrite(file_write_name_quad6,quad6)
    msk_file_write_name_quad6 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-6.png")
    cv2.imwrite(msk_file_write_name_quad6, msk_quad6)

    quad7 = img[1000:1500, 0:500]
    msk_quad7 = msk[1000:1500, 0:500]
    file_write_name_quad7 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-7.png")
    cv2.imwrite(file_write_name_quad7,quad7)
    msk_file_write_name_quad7 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-7.png")
    cv2.imwrite(msk_file_write_name_quad7, msk_quad7)

    quad8 = img[1000:1500, 500:1000]
    msk_quad8 = msk[1000:1500, 0:500]
    file_write_name_quad8 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-8.png")
    cv2.imwrite(file_write_name_quad8,quad8)
    msk_file_write_name_quad8 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-8.png")
    cv2.imwrite(msk_file_write_name_quad8, msk_quad8)

    quad9 = img[1000:1500, 1000:1500]
    msk_quad9 = msk[1000:1500, 1000:1500]
    file_write_name_quad9 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-9.png")
    cv2.imwrite(file_write_name_quad9,quad9)
    msk_file_write_name_quad9 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-9.png")
    cv2.imwrite(msk_file_write_name_quad9, msk_quad9)



def call_cropper():
    mask_src_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\output")
    img_src_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\input")
    img_dest_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_9quad\Images")
    msk_dest_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_9quad\Masks")
    
    count = 0
    for x in mask_src_path.iterdir():
        for y in range(0,2):
            count =count +1
            img_path = img_src_path.joinpath(x.name)
            crop_copy_img(x, img_path, img_dest_path, msk_dest_path)

# call_cropper()

def remove_blank_images():
    mask_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Masks")
    img_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Images")
    dump_path_img = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Images_dump")
    dump_path_mask = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Masks_dump")

    for x in mask_path.iterdir(): 
        img = cv2.imread(str(x),0)
        (unique, counts) = np.unique(img, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        if(frequencies[1][1]) <= 2000:            
            dump_file_mask = dump_path_mask.joinpath(x.name)
            dump_file_img = dump_path_img.joinpath(x.name)
            src_file_img = img_path.joinpath(x.name)
            shutil.move(str(x), str(dump_file_mask))
            shutil.move(str(src_file_img), str(dump_file_img))
            print(x.name)


def bin_images():
    mask_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_9quad\Masks")
    # count = 0
    for x in mask_path.iterdir():
        mg = cv2.imread(str(x),0)
        (unique, counts) = np.unique(mg, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies[1][1])
    

def crop_copy_img_4(msk_path,img_path,img_dest_path,msk_dest_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
    msk = cv2.imread(str(msk_path),0)
    msk = cv2.resize(msk, (1000,1000), interpolation = cv2.INTER_AREA)
    msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    print(img.shape)

    quad1 = img[0:500, 0:500]
    msk_quad1 =  msk[0:500, 0:500]
    (unique_quad1, counts_quad1) = np.unique(msk_quad1, return_counts=True)
    frequencies_quad1 = np.asarray((unique_quad1, counts_quad1)).T
    if(frequencies_quad1[1][1]) <= 1000:  
        file_write_name_quad1 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-1.png")
        cv2.imwrite(file_write_name_quad1,quad1)
        msk_file_write_name_quad1 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-1.png")
        cv2.imwrite(msk_file_write_name_quad1, msk_quad1)

    quad2 = img[0:500, 500:1000]
    msk_quad2 = msk[0:500, 500:1000]
    (unique_quad2, counts_quad2) = np.unique(msk_quad2, return_counts=True)
    frequencies_quad2 = np.asarray((unique_quad2, counts_quad2)).T
    if(frequencies_quad2[1][1]) <= 1000: 
        file_write_name_quad2 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-2.png")
        cv2.imwrite(file_write_name_quad2,quad2)
        msk_file_write_name_quad2 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-2.png")
        cv2.imwrite(msk_file_write_name_quad2, msk_quad2)

    quad3 = img[500:1000, 0:500]
    msk_quad3 = msk[500:1000, 0:500]
    (unique_quad3, counts_quad3) = np.unique(msk_quad3, return_counts=True)
    frequencies_quad3 = np.asarray((unique_quad3, counts_quad3)).T
    if(frequencies_quad3[1][1]) <= 1000: 
        file_write_name_quad3 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-3.png")
        cv2.imwrite(file_write_name_quad3,quad3)
        msk_file_write_name_quad3 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-3.png")
        cv2.imwrite(msk_file_write_name_quad3, msk_quad3)

    quad4 = img[500:1000, 500:1000]
    msk_quad4 = msk[500:1000, 500:1000]
    (unique_quad4, counts_quad4) = np.unique(msk_quad4, return_counts=True)
    frequencies_quad4 = np.asarray((unique_quad4, counts_quad4)).T
    if(frequencies_quad4[1][1]) <= 1000: 
        file_write_name_quad4 = str(str(img_dest_path)  + "\\" + str(img_path.stem) + "-quad-4.png")
        cv2.imwrite(file_write_name_quad4,quad4)
        msk_file_write_name_quad4 = str(str(msk_dest_path)  + "\\" + str(msk_path.stem) + "-quad-4.png")
        cv2.imwrite(msk_file_write_name_quad4, msk_quad4)


def call_4cropper():
    mask_src_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\output")
    img_src_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\input")
    img_dest_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Images")
    msk_dest_path = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\comb_crop_4quad\Masks")
    
    count = 0
    for x in mask_src_path.iterdir():
        img_path = img_src_path.joinpath(x.name)
        crop_copy_img_4(x, img_path, img_dest_path, msk_dest_path)


# call_4cropper()
if __name__ == "__main__":
    call_4cropper()