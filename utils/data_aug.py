import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
from pathlib import Path

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


def augment_data(msk_path):
    img_path = Path('C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\training\input').joinpath(msk_path.name)
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(msk_path), cv2.cv2.IMREAD_GRAYSCALE)

    ori_img_path = Path('C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\training_modified\input').joinpath(msk_path.name)
    cv2.imwrite(str(ori_img_path),image)
    # Horizontal Flip
    hf = A.HorizontalFlip(p=1)
    hf_1 = hf(image=image, mask=mask)
    img_write_path = str(ori_img_path.parent)+ '\\' + str(img_path.stem) + "-hf.png"
    msk_write_path = str(msk_path.parent)+ '\\' + str(msk_path.stem) + "-hf.png"
    cv2.imwrite(img_write_path,hf_1['image'])
    cv2.imwrite(msk_write_path,hf_1['mask'])
    # visualize(hf_1['image'], hf_1['mask'],image, mask)

    # Vertical Flip
    vf = A.VerticalFlip(p=1)
    vf_1 = vf(image=image, mask=mask)
    img_write_path_vf = str(ori_img_path.parent)+ '\\' + str(img_path.stem) + "-vf.png"
    msk_write_path_vf = str(msk_path.parent)+ '\\' + str(msk_path.stem) + "-vf.png"
    cv2.imwrite(img_write_path_vf,vf_1['image'])
    cv2.imwrite(msk_write_path_vf,vf_1['mask'])

    # Random Rotate 90
    rr = A.RandomRotate90(p=1)
    rr_1 = rr(image=image, mask=mask)
    img_write_path_rr = str(ori_img_path.parent)+ '\\' + str(img_path.stem) + "-rr.png"
    msk_write_path_rr = str(msk_path.parent)+ '\\' + str(msk_path.stem) + "-rr.png"
    cv2.imwrite(img_write_path_rr,rr_1['image'])
    cv2.imwrite(msk_write_path_rr,rr_1['mask'])

    # Transpose
    tr = A.Transpose(p=1)
    tr_1 = tr(image=image, mask=mask)
    img_write_path_tr = str(ori_img_path.parent)+ '\\' + str(img_path.stem) + "-tr.png"
    msk_write_path_tr = str(msk_path.parent)+ '\\' + str(msk_path.stem) + "-tr.png"
    cv2.imwrite(img_write_path_tr,tr_1['image'])
    cv2.imwrite(msk_write_path_tr,tr_1['mask'])

    # Shift Scale Rotate
    ss = A.ShiftScaleRotate(p=1)
    ss_1 = ss(image=image, mask=mask)
    img_write_path_ss = str(ori_img_path.parent)+ '\\' + str(img_path.stem) + "-ss.png"
    msk_write_path_ss = str(msk_path.parent)+ '\\' + str(msk_path.stem) + "-ss.png"
    cv2.imwrite(img_write_path_ss,tr_1['image'])
    cv2.imwrite(msk_write_path_ss,tr_1['mask'])

msk_dir = Path('C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\training_modified\output')
for x in msk_dir.iterdir():
    augment_data(Path(x))