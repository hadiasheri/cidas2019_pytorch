"""
    The implementation of {your network's name}
    Author: 
    Date created: March 2019
    Python version: 3.7
    Pytorch version: 1.0.0
"""

# import the required modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms, models
import time
import os

# set the hyper-parameters
batch_size = 20
ler_rate = 0.005
num_epochs = 10
steps = 6

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("We are going to use the GPU ...")

# fixing seeds for reproducibility
torch.manual_seed(0)
if use_gpu:
    torch.cuda.manual_seed(0)

######################################################################
# Loading Data
# ---------

# Data augmentation (only for the training dataset) and normalization
data_transforms = {
    'train': transforms.Compose([  # Composes several transforms together
        transforms.RandomCrop(28),  # data augmentation only for training
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # Convert images to tensors
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
}

# Load the data from the image directory
data_dir = 'R-MNIST'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                        data_transforms[x]) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


# Data Loader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=batch_size,
                                              num_workers=5)
               for x in ['train', 'test']}


######################################################################
# Modules and model specification
# -------------------------------

# CNN Model (2-layer ConvNet)
class CNN(nn.Module):
    def __init__(self, num_out=10):
        super(CNN, self).__init__()
        self.num_out = num_out
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, self.num_out)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

######################################################################
# Train the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase (here, test phase)
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.2f}'.format(
                phase, epoch_loss, epoch_acc * 100))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    # report the training phase
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:2f}'.format(best_acc * 100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

######################################################################
# Evaluation
# ------------------

def eval(model, img):
    model.train(False)  # Set model to evaluate mode
    outputs = model(img)
    _, preds = torch.max(outputs.data, 1)
    return preds

######################################################################
# The main part
# ---------

if __name__ == "__main__":

    model = CNN()
    # model = models.resnet18(pretrained=True)

    if use_gpu:
        model = model.cuda()

    # Weight initializer
    model_parameters = list(model.parameters())
    [init.normal_(m) for m in model_parameters]  # random weight initialization

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=ler_rate)

    # optimizer = optim.Adam([
    #                 {'params': model.layer1.parameters()},
    #                 {'params': model.layer2.parameters()},
    #                 {'params': model.layer2.parameters(), 'lr': 1e-1}
    #                 ], lr=ler_rate)

    # Decay LR by a factor of 0.1 every 8 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)

######################################################################
# Train and save
# ------------------

    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    # Save the trained model
    torch.save(model, 'model.pt')
    # torch.save(model.state_dict(), 'model_parameters.pkl')

######################################################################
# Load the trained model and evaluate it
# ----------------------

    # For just loading a trained model:
    model = torch.load('model.pt')
    # model.load_state_dict(torch.load('model_parameters.pkl'))

    img, target = next(iter(dataloaders['test']))

    lbl = eval(model, img[0].unsqueeze(1))
    print('\nThe true lable: ', target[0].item())
    print('The classifier lable: ', lbl.item())