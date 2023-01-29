import os
import logging as log
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from SDOBenchmark import SDOBenchmarkDataset_time_steps
from torch import nn as nn
from torchvision import models, transforms
from byol import BYOL

from utils import *

# ======================================================================================================================


parser = argparse.ArgumentParser(description="...")

parser.add_argument("--config_path", default="./config.yaml", type=str, help="The config file path")
parser.add_argument("--out_file", default = "./out10.txt", type=str, help = "Output file")
parser.add_argument("--save_weights", default ="./weights4", type = str, help = "Path where model weights are saved")
parser.add_argument('--alpha', default = 1, type = float, help = 'weight of classification loss')
parser.add_argument('--max_epoch', default = 4, type = int, help = 'max trained epoch')

args = parser.parse_args()



# ======================================================================================================================
def GetLabels(labels):
    labb = []
    for i in labels:
        if i > 1e-4:
            labb.append(3)
        elif i > 1e-5:
            labb.append(2)
        elif i > 1e-6:
            labb.append(1)
        else:
            labb.append(0)
    return labb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def toDevice(model):
    return model.to(device)

def initLearner():
    resnet = models.resnet18()
    #change the number of input channels to 4
    resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride = (2, 2), padding=(3, 3), bias=False)
    learner = BYOL(
        resnet,
        image_size=256,
        hidden_layer='avgpool',
        num_channel=4,
        test = True
    )
    learner = toDevice(learner)

    return learner



# ======================================================================================================================

def run(parameters):
    channel = parameters["data"]["channel"]

    if not os.path.isdir(args.save_weights):
        os.mkdir(args.save_weights)

    # model initialisation

    learner = initLearner()
    print(learner)

    # dataset initialisation

    path = Path(parameters["data"]["path"])
    size = parameters["data"]["size"]
    target_transform = flux_to_class_builder(parameters["data"]["targets"]["classes"])
    time_steps = parameters["data"]["time_steps"]

    dataset = SDOBenchmarkDataset_time_steps(
        path / 'test' / 'meta_data.csv',
        path / 'test',
        target_transform=target_transform,
        channel=channel,
        time_steps=time_steps,
        non_flare_only=False,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    epoch = 22

    learner.load_state_dict(torch.load(Path(args.save_weights) / 'model_{}.pth'.format(epoch), map_location=torch.device('cpu')))

    learner.eval()

    Data = []
    Labels = []

    with torch.no_grad():
        for img, img1, img2, label, label1 in dataloader:
            img, img1, img2, label, label1 = img.to(device), img1.to(device), img2.to(device), label.to(device), label1.to(device)

            loss, loss_cl, y, out, representation = learner(img, img1, img2, label)

            Data.append(representation.reshape((-1)).detach().cpu().numpy())
            Labels.append(label1.detach().item())


     # TSNE visualisation

    Labels = np.asarray(GetLabels(Labels))
    Data  = np.asarray(Data)

    labels = np.unique(Labels)
    print('Data.shape : {}, \t Labels.shape: {}, \t unique labels: {}'.format(Data.shape, Labels.shape, labels))

    tsne_model = TSNE(n_components=2, random_state=0)

    Data_embeded = tsne_model.fit_transform(Data)

    for i in labels:
        plt.plot(Data_embeded[Labels==i, 0], Data_embeded[Labels == i, 1], ".", markersize = 5)

    plt.legend(['non-flare', 'C', 'M', 'X'])
    plt.tick_params(axis = 'both', labelsize = 12)
    plt.title('TSNE')
    plt.grid()

    plt.savefig('./tsne3_ep_{}.pdf'.format(epoch))

    plt.close()

if __name__ == '__main__':
    # --- configs and constants ----------------------------------------------------------------------------------------
    with open(args.config_path) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    run(parameters)










