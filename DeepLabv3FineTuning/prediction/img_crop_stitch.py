import cv2, texttable
from pathlib import Path
import numpy as np
from numpy.core.multiarray import concatenate
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import pandas as pd
from matplotlib import pyplot as plt
pd.options.display.float_format = "{:,.4f}".format


def iou_comparison(pred_mask, true_mask, filneame):
    pred_iou = pred_mask.copy()
    true_iou = true_mask.copy()
    pred_iou[pred_iou == 255] = 1
    true_iou[true_iou == 255] = 1
    pred_iou = np.asarray(pred_iou , np.bool)
    true_iou = np.asarray(true_iou, np.bool)
    iou_2d = (np.double(np.bitwise_and(pred_iou, true_iou).sum()) / np.double(np.bitwise_or(pred_iou, true_iou).sum()))
    print("iooooooooou", iou_2d)

    pred_1d = pred_mask.flatten()
    pred_1d[pred_1d == 255] = 1
    
    true_1d = true_mask.flatten()
    true_1d[true_1d == 255] = 1
    
    mse = (np.square(pred_1d - true_1d)).mean(axis=None)
    
    pred_1d = np.asarray(pred_1d, np.bool)
    true_1d = np.asarray(true_1d, np.bool)
    
    iou = (np.double(np.bitwise_and(pred_1d, true_1d).sum()) / np.double(np.bitwise_or(pred_1d, true_1d).sum()))
    iou_j = jaccard_score(pred_1d, true_1d)
    print(iou, iou_j)

def calculate_metrics(pred_mask, true_mask, filneame):
    
    pred_1d = pred_mask.flatten()
    pred_1d[pred_1d == 255] = 1
    
    true_1d = true_mask.flatten()
    true_1d[true_1d == 255] = 1
    
    mse = (np.square(pred_1d - true_1d)).mean(axis=None)
    
    pred_1d = np.asarray(pred_1d, np.bool)
    true_1d = np.asarray(true_1d, np.bool)
    
    iou = (np.double(np.bitwise_and(pred_1d, true_1d).sum()) / np.double(np.bitwise_or(pred_1d, true_1d).sum()))
    iou_j = jaccard_score(pred_1d, true_1d)
    report = classification_report(true_1d,pred_1d, target_names= ['Background', 'Road'], output_dict=True)
    tn, fp, fn, tp = confusion_matrix(true_1d, pred_1d).ravel()

    report['Road'].update({"True Negative":tn, "True Positive":tp, "False Negative":fn, "False Positive":fp, "IOU":round(iou,4), "Accuracy":round(report['accuracy'],4)})
    report.pop('weighted avg')
    report.pop('macro avg')
    report.pop('accuracy')
    print(type(report))
    for key, value in report.items():
        for k, v in value.items():
            value[k] = round(v,4)
    print(report)

    print(filneame)
    print(pd.DataFrame.from_dict(report))
    

    # # print(table.draw())
    # print(tn, fp, fn, tp)
    # print(iou, iou_j)
    # print("{} has IOU of {}".format(str(filneame),iou_j))
    # print("Following is the Precision Recall F1-score and accuracy for {}\n {}".format(str(filneame),report))
    # print(str(filneame), mse)
    # print(type(report))
    return report
    

pred_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\pred_7x7\img-13.png")
true_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\\true\img-13.png")
pred_mask = cv2.imread(str(pred_path))
true_mask = cv2.imread(str(true_path))

calculate_metrics(pred_mask, true_mask,pred_path.name)
# iou_comparison(pred_mask, true_mask,pred_path.name)

def get_training_file_loss(pred_mask, true_mask):

    pred_1d = pred_mask.flatten()
    pred_1d[pred_1d == 255] = 1
    
    true_1d = true_mask.flatten()
    true_1d[true_1d == 255] = 1
    
    mse = (np.square(pred_1d - true_1d)).mean(axis=None)
    return mse, np.sum(true_1d), (mse/np.sum(true_1d))

# pred_dir = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\pred_7x7")
# true_dir = Path("C:\\Users\\nbhas\Desktop\Shishir\\road_segmentation_ideal\\road_segmentation_ideal\\training\output")

def calc_save_loss():
    mse_list = []

    for x in true_dir.iterdir():
        true_mask = cv2.imread(str(x))
        pred_mask = cv2.imread(str(pred_dir.joinpath(x.name)))
        mse, count, ratio = get_training_file_loss(pred_mask, true_mask)
        print(str(x.name), mse)
        mse_list.append([str(x.name), mse, count, ratio])
    print(mse_list)

    my_df = pd.DataFrame(mse_list)
    my_df.to_csv('loss.csv', index=False, header=['filename', 'mse', 'count', 'ratio'])

# calc_save_loss()

def get_edges():
    true_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\\true")
    true_edge_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\\true_edges")
    pred_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\pred_7x7")
    pred_edge = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\pred_7x7_edges")
    for x in true_path.iterdir():
        true_img = cv2.imread(str(x))
        true_edges = cv2.Canny(true_img,100,200)
        cv2.imwrite(str(true_edge_path.joinpath(x.name)), true_edges)

        pred_img = cv2.imread(str(pred_path.joinpath(x.name)))
        pred_edges = cv2.Canny(pred_img,100,200)
        cv2.imwrite(str(pred_edge.joinpath(x.name)), pred_edges)

    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

# get_edges()