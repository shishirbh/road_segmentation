import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from torch import set_flush_denormal
from model import createDeepLabv3
import numpy as np
from pathlib import Path

# Load the trained model 
checkpoint = torch.load('C:\\Users\\nbhas\Desktop\\Shishir\\DeepLabv3FineTuning\\CFExp\\epoch-complete-12.pth')
model = createDeepLabv3()
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(checkpoint['state_dict'])
# Set the model to evaluate mode
model.eval()


def crop_copy_img_4(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
    print(img.shape)
    quad1 = img[0:500, 0:500]
    quad2 = img[0:500, 500:1000]
    quad3 = img[500:1000, 0:500]
    quad4 = img[500:1000, 500:1000]
 
    return quad1, quad2, quad3, quad4 #, msk_quad1, msk_quad2, msk_quad3, msk_quad4

def stich_and_upscale(msk_quad1, msk_quad2, msk_quad3, msk_quad4, filename):
    top_half = np.concatenate((msk_quad1, msk_quad2), axis=1)
    bottom_half = np.concatenate((msk_quad3, msk_quad4), axis=1)
    comp_pred_mask = np.concatenate((top_half, bottom_half),axis=0)
    complete_pred_mask = cv2.resize(comp_pred_mask,(1500,1500))
    # complete_pred_mask = post_process_output(complete_pred_mask)
    out_msk_file = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\output\pred").joinpath(filename)
    cv2.imwrite(str(out_msk_file), complete_pred_mask)

def post_process_output(pred_mask):
        kernel = np.ones((3,3),np.uint8)
        return cv2.erode(pred_mask,kernel,iterations = 1)

class RoadSegModel:

    def __init__(self, img, msk_path):
        """
        docstring
        """
        self.img = img
        self.msk_path = msk_path
        pass

    def get_pred(self):
        
        ino = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = self.img.transpose(2,0,1).reshape(1,3,500,500)
        with torch.no_grad():
            a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
        print(a)
        b= a['out']
        b[b < 0.1] = 0
        b[b >= 0.1] = 255
        # print(type(b.cpu().detach().numpy()))
        # print(b[0][0].shape)
        b=b.cpu().detach().numpy()
        # print(b[0][0])
        pred_mask = (b[0][0])

        # pred_mask = cv2.cvtColor(b[0][0], cv2.COLOR_BGR2GRAY)
        cv2.imwrite("test1.png",pred_mask)
        # plt.hist(a['out'].data.cpu().numpy().flatten())
        return pred_mask


def run_test(img_dir):

    for img_path in img_dir.iterdir():
        # img_path = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\input\img-1.png")

        quad1, quad2, quad3, quad4 = crop_copy_img_4(str(img_path))
        mq1 = RoadSegModel(quad1, None)
        print(mq1.get_pred())
        msk_quad1 = mq1.get_pred()

        mq2 = RoadSegModel(quad2, None)
        print(mq2.get_pred())
        msk_quad2 = mq2.get_pred()

        mq3 = RoadSegModel(quad3, None)
        print(mq3.get_pred())
        msk_quad3 = mq3.get_pred()

        mq4 = RoadSegModel(quad4, None)
        print(mq4.get_pred())
        msk_quad4 = mq4.get_pred()

        stich_and_upscale(msk_quad1, msk_quad2, msk_quad3, msk_quad4, img_path.name)

img_dir = Path("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\prediction\input")
run_test(img_dir)