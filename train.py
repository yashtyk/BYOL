import argparse
import os
from pathlib import Path
import torch
from torchvision import models
from byol import BYOL
import torch.nn as nn
import yaml
import random

import math

from utils import flux_to_class_builder
from SDOBenchmark import SDOBenchmarkDataset_time_steps
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser(description="...")

parser.add_argument("--config_path", default="./config.yaml", type=str, help="The config file path")
parser.add_argument("--out_file", default = "./out.txt", type=str, help = "Output file")
parser.add_argument("--save_weights", default ="./weights", type = str, help = "Path to folder where model weights will be saved")
parser.add_argument('--alpha', default = 1, type = float, help = 'weight of classification loss')
parser.add_argument('--epoch_pretr', default = -1, type = int, help = 'number of pretrained model')
parser.add_argument('--pretr_path', default = './weights/model.pth', type = str, help = 'path to pretrained weights' )

args = parser.parse_args()

#device

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
        num_channel=4
    )
    learner = toDevice(learner)

    return learner



def train_test_split(dataset, test_size = 0.1, seed = 0):
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be in [0, 1].")

    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size



    generator = torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)



def run(parameters):

    channel = parameters ["data"]["channel"]

    if not os.path.isdir(args.save_weights):
        os.mkdir(args.save_weights)

    # model initialisation

    learner = initLearner()
    print(learner)

    lr = 0.005
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, 300, 0.0001)
    # dataset initialisation

    path = Path (parameters["data"]["path"])
    size = parameters["data"]["size"]
    target_transform = flux_to_class_builder(parameters["data"]["targets"]["classes"])
    time_steps = parameters["data"]["time_steps"]

    dataset = SDOBenchmarkDataset_time_steps(
        path / 'training' / 'meta_data.csv',
        path / 'training',
        target_transform = target_transform,
        channel = channel,
        time_steps = time_steps,
        non_flare_only= False,
    )

    seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)



    dataset_tr, dataset_val = train_test_split(dataset)

    dataloader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=200, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=200, shuffle=True)

    learner.target_ema_updater.beta = 0.996

    epochs = 300
    alpha = args.alpha

    f = open(Path(args.out_file), 'a')
    f.write('training on {}, lr: {}, weights_saved: {} , alpha: {}\n, val_seed: {}'.format(parameters["data"]["channel"], lr, args.save_weights, args.alpha, seed))
    f.close()

    # create directory to save weights
    if not os.path.isdir(Path(args.save_weights) ):
        os.mkdir(Path(args.save_weights))

    #training
    # load weights
    if args.epoch_pretr >=0:
        learner.load_state_dict(torch.load(args.pretr_path))

        for epoch in range (0, args.epoch_pretr+1):
            scheduler.step()

    start = (args.epoch_pretr + 1) if args.epoch_pretr >=0 else 0
    for epoch in range(start,  epochs):
        learner.train()
        learner.target_ema_updater.beta = float(1 - (1 - 0.996) * (math.cos(math.pi * epoch / 300) + 1) / 2)
        loss_tr = 0
        count = 0
        print(epoch)
        for img, img1, img2, label in dataloader_tr:
            img, img1, img2, label= img.to(device), img1.to(device), img2.to(device), label.to(device)

            loss, loss_cl = learner( img,img1, img2,  label)
            loss_all = loss + alpha* loss_cl
            opt.zero_grad()
            loss_all.backward()
            opt.step()
            loss_tr += loss_all.detach()
            count += 1
            learner.update_moving_average()  # update moving average of target encoder

        torch.save(learner.state_dict(), Path(args.save_weights) / 'model_{}.pth'.format(epoch))
        scheduler.step()
        # validation
        with torch.no_grad():
            loss_vall = 0
            count_vall = 0
            learner.eval()
            for img, img1, img2, label in dataloader_val:
                img, img1, img2, label= img.to(device), img1.to(device), img2.to(device), label.to(device)

                loss, loss_cl = learner( img,img1, img2,  label)


                loss_all = alpha * loss_cl
                loss_vall += loss_all.detach()
                count_vall+=1
            f = open(Path(args.out_file), 'a')
            f.write('epoch: {} , loss: {:.4f}, loss_val : {:4f}, lr: {}\n'.format(epoch, loss_tr / count,
                                                                                  loss_vall / count_vall,
                                                                                  opt.param_groups[0]['lr']))
            f.close()


if __name__ == '__main__':
    # --- configs and constants ----------------------------------------------------------------------------------------
    with open(args.config_path) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    run(parameters)

















