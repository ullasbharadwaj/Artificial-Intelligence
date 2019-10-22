import model
import numpy as np
from collections import deque
import math
import os
import tensorflow as tf
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2
import glob
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable,grad
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import ImageFolder
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import Compose,ToTensor,ToPILImage,Resize,Normalize,CenterCrop


HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 12
EPOCH = 5000
LAMBDA = 10 # Gradient penalty lambda hyperparameter
num_episodes = 5000
model_path = './Models/'
train_path = './Data/'
early_stopping = 7
batch_size = 12

"""
Subroutine for Early Stopping.
"""
def check_early_stopping(val_loss, val_list, extrctr):
    if len(val_list)< val_list.maxlen:
        return
    if True in (val_loss < t for t in val_list):
        return
    else:
        print('Early Stopping')
        create_test_files(extrctr)
        exit()


"""
Subroutine for NN weights initialization.
WGAN need proper initialization of weights
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


"""
Subroutine for saving any image.

Please note to consider the format, if
4 channels are present in the image, use 'png'
"""
def save_image_(img):
    cv2.imwrite('sample_img.png',img)


"""
Reduce the learning rate if the val error increases.
"""
def learning_rate_decay(optimizer, decay = 0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay


"""
Subroutine to clip the gradients and avoid Exploding-Gradients.
"""
def gradient_clipping(dnn_model, clip = 10):
    torch.nn.utils.clip_grad_norm_(dnn_model.parameters(),clip)

"""
Subroutine to penalize the gradients. Exclusive to WGANs.
"""
def loss_with_penalty(D, real_data, fake_data):
    alpha = Variable(Tensor(np.random.uniform(0, 1, real_data.size()).astype(np.float32)))#torch.rand(rdata.size())
    alpha = alpha.to(device)

    Interpolated_data = alpha * real_data + ((1 - alpha) * fake_data)

    Interpolated_data = Variable(Interpolated_data, requires_grad=True)
    Interpolated_data = Interpolated_data.to(device)

    d_interpolated_data = D(Interpolated_data)
    gradients = grad(outputs=d_interpolated_data, inputs=Interpolated_data,
                              grad_outputs=torch.ones(d_interpolated_data.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    # Return the computed Gradient Penalty
    return gradient_penalty


"""
Class for defining the DataLoader
"""

class DNNdataset(Dataset):
    """Return a new dataset."""

    def __init__(
        self,
        train_folder_path,
    ):
        self.data_files = glob.glob(train_folder_path + "/*.png")
        self.train_len = len(self.data_files)
        print(self.train_len)
        self.train_folder_path = train_folder_path
        self.trans= Compose([ToTensor()])#,Normalize((0.5, 0.5, 0.5,1), (0.5, 0.5, 0.5,1))])

    def __len__(self):
        return self.train_len

    def __getitem__(self,idx):
        with open(self.data_files[idx], 'rb' ) as handle:
            sample = np.asarray(Image.open(handle).resize((128,128)))
        return self.trans(sample)#image.eval(session=sess)


availbl = torch.cuda.is_available()availbl = torch.cuda.is_available()
if availbl:
    device = 'cuda'
else:
    device = 'cpu'
if early_stopping > 0:
    early_stop= deque(maxlen=early_stopping)

dataset = DNNdataset(train_path)
dataloaderr = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers=8, drop_last=True)

#Defintion of Input and Output dimensions, Random_Dimension corresponds to the Noise Sequence used to
#generate Fake Images
CHANNEL = 4    #JPG-3, PNG-4
output_dim = CHANNEL
input_dim = 128
random_dim = 128

#Define the Generator model
Generator = model.GAN_Generator(input_dim, output_dim, random_dim, batch_size)
Generator.train()
Generator.to(device)

#Define the Discriminator model
Discriminator = model.GAN_Discrimator(input_dim, output_dim, batch_size)
Discriminator.train()
Discriminator.to(device)

#GANs need to be initialized properly
Generator.apply(weights_init)
Discriminator.apply(weights_init)

#hyperparameter settings defnition
learning_rate = 0.0002
clip_value = 0.01
n_critic = 5
sample_interval = 10
latent_dim = 4
old_score = 1e10

#Setting up the Optimizers for training the GAN
optimizer_D = Adam(Discriminator.parameters(),lr=1e-4,betas=[0.01,.9])
optimizer_G = Adam(Generator.parameters(),lr=1e-4,betas=[0.01,.9])

Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

#Used to control the frquency of updating the weights of Generator
d_iters = 1
g_iters = 1

for episode_i in range(1,num_episodes+1):
counter = 0

    for batchh in dataloaderr:
        counter += 1

        real_image = batchh.float()

        if real_image.shape[1] != 3 and real_image.shape[1] != 4:
            real_image = torch.transpose(real_image,1,3)

        real_image = real_image.float().to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for k in range(d_iters):
            for p in Discriminator.parameters():
                p.requires_grad = True

            optimizer_D.zero_grad()

            train_noise = Variable(Tensor(np.random.uniform(0, 1, (batch_size, random_dim)).astype(np.float32)))# batch_siize x 512 x 4 x 4
            train_noise.to(device)
            fake_image = Generator(train_noise)
            fake_res = (Discriminator(fake_image)).mean()
            real_res = (Discriminator(real_image)).mean()

            w_loss = loss_with_penalty(Discriminator,real_image.data,fake_image.data)
            loss_D = -real_res + fake_res + w_loss

            loss_D.backward()

            optimizer_D.step()
        # Clip weights of discriminator
        # for p in Discriminator.parameters():
        #     p.data.clamp_(-clip_value, clip_value)

        # -----------------
        #  Train Generator
        # -----------------

        for k in range(g_iters):
            for p in Generator.parameters():
                p.requires_grad = True

            optimizer_G.zero_grad()

            train_noise = Variable(Tensor(np.random.uniform(0, 1, (batch_size, random_dim)).astype(np.float32)))# batch_siize x 512 x 4 x 4
            train_noise.to(device)
            gen_image = Generator(train_noise)
            gen_score = Discriminator(gen_image)
            loss_Gen = -gen_score.mean()

            loss_Gen.backward()

            optimizer_G.step()

            print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (episode_i, num_episodes, counter % len(dataloaderr), len(dataloaderr), loss_D.item(), loss_Gen.item()),end='\r')

    #Save the generated fake image every 10th epoch
    if episode_i % sample_interval == 0:
        gen_image=gen_image.cpu().data * 1.0#0.5 + 0.5
        img = ToPILImage()(gen_image[10])
        img.save("generated_images/ep{}.png".format(episode_i))

    #Save the Model and State Dictionary every 200th epoch
    if episode_i % 200== 0:
        torch.save(Generator.state_dict(), '{}/GAN_ep{}loss{:.5f}.pth'.format(model_path, episode_i, loss_Gen))
        print('\nsaved model state dict\n')

    #Code for early stopping - Not used currently
    # if loss_G > old_score:
    #     learning_rate_decay(optimizer)
    #     print('new leanring rate')
    # else:
    #     old_score = loss_G
