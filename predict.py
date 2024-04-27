import argparse

import torch
from PIL import Image
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import transforms

from model.SiameseNetwork import SiameseNetwork


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', '-i1', action='store', required=True, help="original image")
    parser.add_argument('--image2', '-i2', action='store', required=True, help="detect image")
    args = parser.parse_args()
    return args


def eu_distance(net, img0, img1):
    output1, output2 = net(img0, img1)
    e_distance = F.pairwise_distance(output1, output2)
    return e_distance


def get_prob(eu_dis):
    return nn.Sigmoid(eu_dis)


if __name__ == "__main__":

    arg = create_arg_parser()

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model = SiameseNetwork()

    model.load_state_dict(torch.load('model/weights/model_siamesenet_prob_2.pth'))
    model.to(device)
    data_transforms = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])

    image_1 = Image.open(arg.image1)
    try:
        image_1 = data_transforms(image_1.convert('L')).unsqueeze(0)
    except:
        print('Image_1 Open Error! Try again!')

    image_2 = Image.open(arg.image2)
    try:
        image_2 = data_transforms(image_2.convert('L')).unsqueeze(0)
    except:
        print('Image_2 Open Error! Try again!')
    model.eval()
    output = model(image_1.to(device), image_2.to(device))
    log_probas = F.sigmoid(output)
    print(log_probas.item())



