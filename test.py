import argparse
import os
from pathlib import Path
import torch
from torchvision import models
from byol import BYOL
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix



from utils import flux_to_class_builder
from SDOBenchmark import SDOBenchmarkDataset_time_steps

parser = argparse.ArgumentParser(description="...")

parser.add_argument("--config_path", default="./config.yaml", type=str, help="The config file path")
parser.add_argument("--out_file", default = "./out9.txt", type=str, help = "Output file")
parser.add_argument("--save_weights", default ="./weights3", type = str, help = "Path where model weights are saved")
parser.add_argument('--alpha', default = 1, type = float, help = 'weight of classification loss')
parser.add_argument('--num_epoch', default = [1], type = list, help = 'list of numbers of trained epochs'  )

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
        num_channel=4,
        test = True
    )
    learner = toDevice(learner)

    return learner

def tss(tp: int, fp: int, tn: int, fn: int) -> float:
    # True Skill Statistic
    # also computed as sensitivity + specificity - 1
    # in [-1,1], best at 1, no skill at 0
    # Always majority class: 0
    # Random: 0
    return tp / (tp + fn ) + tn / (tn + fp ) - 1

def run(parameters):

    channel = parameters ["data"]["channel"]

    if not os.path.isdir(args.save_weights):
        os.mkdir(args.save_weights)

    # model initialisation

    learner = initLearner()
    print(learner)

    # dataset initialisation

    path = Path (parameters["data"]["path"])
    size = parameters["data"]["size"]
    target_transform = flux_to_class_builder(parameters["data"]["targets"]["classes"])
    time_steps = parameters["data"]["time_steps"]

    dataset = SDOBenchmarkDataset_time_steps(
        path / 'test' / 'meta_data.csv',
        path / 'test',
        target_transform = target_transform,
        channel = channel,
        time_steps = time_steps,
        non_flare_only= False,
    )



    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)





    alpha = args.alpha

    f = open(Path(args.out_file), 'a')
    f.write('test on {}, weights_saved: {} , alpha: {}\n'.format(parameters["data"]["channel"], args.save_weights, args.alpha))
    f.close()




    #test

    for epoch in args.num_epoch:
        learner.load_state_dict(torch.load(Path(args.save_weights) / 'model_{}.pth'.format(epoch), map_location=torch.device('cpu')))



        learner.eval()

        loss_test = 0
        count = 0
        y_all = []
        y_out = []
        y_pred  =[]
        with torch.no_grad():
            for img, img1, img2, label in dataloader:
                img, img1, img2, label= img.to(device), img1.to(device), img2.to(device), label.to(device)

                loss, loss_cl, y , out = learner( img,img1, img2,  label)
                y_all += label.int().tolist()
                y_pred += torch.argmax(out, dim=1).tolist()
                y_out += out.tolist()
                loss_all = loss + alpha* loss_cl
                loss_test += loss_all.detach()
                count += 1

        # calculate tss
        print(type(y_all))

        tn, fp, fn, tp = confusion_matrix(y_all, y_pred ).ravel()
        tss_value = tss(tp, fp, tn, fn)



        f = open(Path(args.out_file), 'a')
        f.write( 'epoch: {},test_loss: {:.4f}, tn: {}, fp: {}, fn: {}, tp: {}, tss: {:.4f}, '.format(epoch,  loss_test / count, tn, fp, fn, tp, tss_value))
        f.close()



if __name__ == '__main__':
    # --- configs and constants ----------------------------------------------------------------------------------------
    with open(args.config_path) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    run(parameters)


