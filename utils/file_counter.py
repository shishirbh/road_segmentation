from pathlib import Path
import shutil
# img_path = Path('C:\\Users\\nbhas\Desktop\\Shishir\\models-master\\research\\deeplab\\datasets\\pascal_voc_seg\\VOCdevkit\\VOC2012\\comb')
# seg_path = Path("C:\\Users\\nbhas\\Desktop\\Shishir\\models-master\\research\\deeplab\\datasets\\pascal_voc_seg\\VOCdevkit\\VOC2012\\SegmentationClassRaw")
filter_img_path = Path('C:\\Users\\nbhas\Desktop\\Shishir\\models-master\\research\\deeplab\\datasets\\pascal_voc_seg\\VOCdevkit\\VOC2012\\JPEGImages')

# seg_path_lst =[]
# for x in seg_path.iterdir():
#     seg_path_lst.append(x)
#     # print(x)
# print(len(seg_path_lst))

# img_path_lst =[]
# for x in img_path.iterdir():
#     img_path_lst.append(x)
#     # print(x)
# print(len(img_path_lst))

# for x in seg_path_lst:
#     # print(x.name)
#     # target_path = str(filter_img_path) + "\\" + str(x.name)
#     source_path = img_path.joinpath(x.name)
#     target_path = filter_img_path.joinpath(x.name)
#     shutil.copy2(str(source_path),str(target_path))
#     # print(target_path)

filt_img_path_lst =[]
for x in filter_img_path.iterdir():
    filt_img_path_lst.append(x)
    print(x.stem)
# print((filt_img_path_lst))
