import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from model import createDeepLabv3
from trainer import train_model
import datahandler
import argparse
import os
import torch

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--batchsize", default=2, type=int)

args = parser.parse_args()

bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

##############
# checkpoint = torch.load("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\exp\epoch-complete-4.pth")
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# epochs = checkpoint['epoch']
# # loss = checkpoint['loss']
##############
##############
# checkpoint = torch.load("C:\\Users\\nbhas\Desktop\Shishir\DeepLabv3FineTuning\exp\epoch-complete-4.pth")
# pretrained_dict = checkpoint['state_dict']
# model_dict = model.state_dict()
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict) 
# # 3. load the new state dict
# model.load_state_dict(model_dict)
# optimizer.load_state_dict(checkpoint['optimizer'])
# epochs = checkpoint['epoch']
# loss = checkpoint['loss']
##############

model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


if __name__ == "__main__":
    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    

    # Specify the evalutation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}


    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_dir, batch_size=batchsize)
    trained_model = train_model(model, criterion, dataloaders,
                                optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


    # Save the trained model
    # torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
    torch.save(model, os.path.join(bpath, 'weights.pt'))