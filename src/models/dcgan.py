from torch import flatten
from torch import nn

class Generator(nn.Module):
    def __init__(self, inputDim=100, outputDim=512, outputChannels=1):
        super(Generator, self).__init__()

        # first set of CONVT => RELU => BN.
        self.ct1 = nn.ConvTranspose2d(
            in_channels=inputDim,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False
        )
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(128)

        # second set of CONVT => RELU => BN.
        self.ct2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(64)

        # last set of CONVT => RELU => BN.
        self.ct3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.relu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(32)

        # apply upsample and transposed convulition,
        # but use Tanh as the activation function.
        self.ct4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=outputChannels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batchNorm1(x)

        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)

        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchNorm3(x)

        x = self.ct4(x)
        output = self.tanh(x)

        return output

class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super(Discriminator, self).__init__()

        # first set of CONV => RELU layers.
        self.conv1 = nn.Conv2d(
            in_channels=depth,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.leakyRelu1 = nn.LeakyReLU(alpha, inplace=True)

        # second set of CONV => RELU layers.
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.leakyRelu2 = nn.LeakyReLU(alpha, inplace=True)

        # single set of FC => RELU layers.
        self.fc1 = nn.Linear(
            in_features=3136,
            out_features=512
        )
        self.leakyRelu3 = nn.LeakyReLU(alpha, inplace=True)

        # sigmoid layer to output a single value.
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=1
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyRelu1(x)

        x = self.conv2(x)
        x = self.leakyRelu2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu3(x)

        x = self.fc2(x)
        output = self.sigmoid(x)

        return output


# Training part.

from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
import numpy as np
import argparse
import torch
import cv2
import os

# custom weights initialization.
def weights_init(model):
    # get the class name
    classname = model.__class__.__name__

    # check if the classname contains the word "Conv".
    if classname.find("Conv") != -1:
        # initialize the weights from normal distribution.
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    # otherwise, check if the name contains the word "BatchNorm".
    elif classname.find("BatchNorm") != -1:
        # initialize the weights from normal distribution and set bias to 0.
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to output directory.")
ap.add_argument("-e", "--epochs", type=int, default=20,
    help="number of epochs to train for.")
app.add_argument("-b", "--batch-size", type=int, default=128,
    help="size of the batches.")
args = vars(ap.parse_args())

# store epochs and batch size in constant variables.
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

# set the device.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define data transforms.
dataTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))]
)

# load dataset.
print("[INFO] loading dataset...")
trainData = MNIST(root="data", train=True, download=True,
    transform=dataTransforms)
testData = MNIST(root="data", train=False, download=True,
    transform=dataTransforms)
data = torch.utils.data.ConcatDataset([trainData, testData])

# initialize dataloader.
dataloader = DataLoader(data, shuffle=True,
    batch_size=BATCH_SIZE)

# calculate steps per epoch
stepsPerEpoch = len(dataloader.dataset) // BATCH_SIZE

# build generator, initialize weights, and set it to the device.
print("[INFO] building generator...")
gen = Generator(inputDim=100, outputDim=512, outputChannels=1)
gen.apply(weights_init)
gen.to(DEVICE)

# build discriminator, initialize weights, and set it to the device.
print("[INFO] building discriminator...")
disc = Discriminator(depth=1, alpha=0.2)
disc.apply(weights_init)
disc.to(DEVICE)

# construct the optimizer for both generator and discriminator.
genOpt = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999),
    weight_decay=0.0002 / NUM_EPOCHS)
discOpt = Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999),
    weight_decay=0.0002 / NUM_EPOCHS)

# initialize BCELoss function.
criterion = nn.BCELoss()

print("[INFO] starting training...")
benchmarkNoise = torch.randn(256, 100, 1, 1, device=DEVICE)

# define real and fake label values
realLabel = 1
fakeLabel = 0

# loop over epochs.
for epoch in range(NUM_EPOCHS):
    # show epoch information and compute number of batches per epoch.
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))

    # initialize current loss for generator and discriminator.
    epochLossG = 0
    epochLossD = 0

    for x in dataloader:
        # zero the discriminator gradients.
        disc.zero_grad()

        # grab images and send them to the device.
        images = x[0]
        images = images.to(DEVICE)

        # get batch size and create a labels tensor
        bs = images.size(0)
        labels = torch.full((bs,), realLabel, dtype=torch.float,
            device=DEVICE)
        
        # forward pass through discriminator.
        output = disc(images).view(-1)

        # calculate loss on all-real batch.
        errorReal = criterion(output, labels)

        # calculate gradients for discriminator in backward pass.
        errorReal.backward()

        # randomly generate noise for the generator.
        noise = torch.randn(bs, 100, 1, 1, device=DEVICE)

        # generate a fake image batch using the generator
        fake = gen(noise)
        labels.fill_(fakeLabel)

        # forward pass through discriminator again with fake batch.
        output = disc(fake.detach()).view(-1)
        errorFake = criterion(output, labels)

        # calculate gradients for discriminator in backward pass.
        errorFake.backward()

        # compute error for discriminator and update weights.
        errorD = errorReal + errorFake
        discOpt.step()

        # zero the generator gradients.
        gen.zero_grad() 

        # update the labels as fake labels are real for the generator
        # perform a forward pass of fake data batch through discriminator again..
        labels.fill_(realLabel)
        output = disc(fake).view(-1)

        # calculate generator's loss based on output from discriminator.
        # calculate gradients for generator and perform an optimizer step.
        errorG = criterion(output, labels)
        errorG.backward()
        genOpt.step()

        # add current iteration loss of discriminator and generator to epoch loss.
        epochLossD += errorD
        epochLossG += errorG

    # display epoch information.
    print("[INFO] Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
        epochLossG / stepsPerEpoch, epochLossD / stepsPerEpoch))
    