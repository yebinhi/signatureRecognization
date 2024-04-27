# import the necessary libraries
import time

import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from config import config
from dataset.dataset import SiameseDataset
from model.SiameseNetwork import SiameseNetwork
from model.losses import BinaryCrossEntropyWithLogits
import torch.nn.functional as F


# train the model
def train(epochs, model, criterion, train_dl, optimizer, scheduler, device):
    losses = []

    for epoch in range(1, epochs):
        print(f'------------- epoch {epoch} ------------- learning rate: {scheduler.get_last_lr()} -------------')
        start = time.time()
        for batch_idx, data in enumerate(train_dl):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            # forward
            output = model(img0, img1)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            losses.append(loss.item())
            # step
            optimizer.step()

        scheduler.step()
        end = time.time()
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}, time elapsed: {end - start}')
    return model


def test(model, test_dataloader):
    count = 0
    for i, data in enumerate(test_dataloader):
        x0, x1, label = data
        output = model(x0.to(device), x1.to(device))
        log_probas = F.sigmoid(output)
        # print("Actual Label: " + str(label[0][0].item()) + ' predicted: ' + str(log_probas.item()))
        if label == 1 and log_probas > 0.5:
            count = count + 1
        elif label == 0 and log_probas < 0.5:
            count = count + 1
    print('percision: '+str(count/test_dataloader.__len__()))


if __name__ == '__main__':
    # load the dataset
    print('start loading config')
    training_dir = config.training_dir
    testing_dir = config.testing_dir
    training_csv = config.training_csv
    testing_csv = config.testing_csv

    # define data transform
    print('defining data transform')
    data_transforms = transforms.Compose([transforms.Resize((105, 105)),
                                          transforms.ToTensor(),
                                          # transforms.Grayscale(num_output_channels=3),
                                          # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                          ])

    # Load the the dataset from raw image folders
    print('Loading the the dataset from raw image folders')
    train_dataset = SiameseDataset(
        training_csv,
        training_dir,
        transform=data_transforms,
    )

    test_dataset = SiameseDataset(
        testing_csv,
        testing_dir,
        transform=data_transforms)

    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=config.batch_size)

    test_dataloader = DataLoader(test_dataset,
                                 shuffle=True,
                                 num_workers=6,
                                 batch_size=1)

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print('program will run on: ' + device)

    # Declare Siamese Network
    model = SiameseNetwork()
    model.load_state_dict(torch.load('model/weights/model_siamesenet_prob_2.pth'))
    # model = SiameseNet()
    # model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    # Declare Loss Function
    # criterion = ContrastiveLoss()
    # criterion = TripletLoss(1)
    criterion = BinaryCrossEntropyWithLogits()

    # Declare Optimizer and other config
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.max_lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.max_lr, weight_decay=0.0005)
    optimizer = optim.RMSprop(model.parameters(), lr=config.max_lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # start training
    print('start training')
    # train(config.epochs, model, criterion, train_dataloader, optimizer, scheduler, device)
    # torch.save(model.state_dict(), 'model/weights/model_siamesenet_prob_2.pth')
    #model.load_state_dict(torch.load('model/weights/model_siamesenet_prob.pth'))
    print("Model Saved Successfully")
    test(model, test_dataloader)
