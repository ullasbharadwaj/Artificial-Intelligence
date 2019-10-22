"""
Python Source file containing the
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

torch.manual_seed(1)

#Information of number of filters in Generator and Discriminator at each Conv/Deconv Layer
gen_channel =[512, 256, 128, 64, 32]
dis_channel = [32,64, 128, 256]
"""
#############################################################################
                Model for Image Generator

    The Image generator takes in the input noise sequences
    and generates a meaningful image using the deconvolution
    operations.

    It comprises of inital Feed Forward Layers, followed
    by set of deconvolution layers with decresing number
    of filters and the last deconvolution layer gives the
    tanh activated image of required dimension.

    ****
    The deconvolution operation:
    The Image resolution goes on increasing after each deconvolution
    and the filter dimension is reduced after each deconvolution
    ****
#############################################################################
"""
class GAN_Generator(nn.Module):
    def __init__(self, input_dim, output_dim, random_dim, batch_size):
        super(GAN_Generator,self).__init__()

        self.output_dim = output_dim
        self.random_dim = random_dim
        self.batch_size = batch_size

        self.initial_img_dim = 4 # Final Image Shape = 128x128x4, Initial Img Shape = 4x4x512 (Note:Channel First Representation is used in the network)

        self.fc1 = nn.Linear(self.random_dim, gen_channel[0] )
        self.fc2 = nn.Linear(gen_channel[0], gen_channel[0] * self.initial_img_dim * self.initial_img_dim)
        self.fc2_norm = nn.BatchNorm1d(gen_channel[0] * self.initial_img_dim * self.initial_img_dim)


        self.conv1 = nn.ConvTranspose2d(gen_channel[0], gen_channel[1], kernel_size = (5,5), stride = 2, padding = 2, output_padding = 1)
        self.norm1 = nn.BatchNorm2d(gen_channel[1])

        self.conv2 = nn.ConvTranspose2d(gen_channel[1], gen_channel[2], kernel_size = (5,5), stride = 2, padding = 2, output_padding = 1)
        self.norm2 = nn.BatchNorm2d(gen_channel[2])

        self.conv3 = nn.ConvTranspose2d(gen_channel[2], gen_channel[3], kernel_size = (5,5), stride = 2, padding = 2, output_padding = 1)
        self.norm3 = nn.BatchNorm2d(gen_channel[3])

        self.conv4 = nn.ConvTranspose2d(gen_channel[3], gen_channel[4], kernel_size = (5,5), stride = 2, padding = 2, output_padding = 1)
        self.norm4 = nn.BatchNorm2d(gen_channel[4])

        self.conv5 = nn.ConvTranspose2d(gen_channel[4], self.output_dim, kernel_size = (5,5), stride = 2, padding = 2, output_padding = 1)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, noise):

        #Noise shape - (batch_size, self.random_dim)
        x = self.fc1(noise)
        x = F.relu(x)

        #Shape - (batch_size, gen_channel[0]) = (batch_size, 512)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2_norm(x)

        #Shape - (batch_size, gen_channel[0]*4*4) = (batch_size, 8192)
        x = x.view(self.batch_size, 512, 4, 4)

        #Shape - (batch_size, 512, 4, 4)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        #Shape - (batch_size, 256, 8, 8)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        #Shape - (batch_size, 128, 16, 16)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        #Shape - (batch_size, 64, 32, 32)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)

        #Shape - (batch_size, 32, 64, 64)
        x = self.conv5(x)
        x = F.tanh(x)
        # Note: No batch normalization should be applied to the last deconvolution layer
        #Shape - (batch_size, 4, 128, 128)
        return x

"""
#############################################################################
                Model for Image Classifier

    The Image Classifier is called as the Discriminator
                   in the GAN network.
#############################################################################
"""
class GAN_Discrimator(nn.Module):

    def __init__(self, input_dim, output_dim, batch_size):
        super(GAN_Discrimator,self).__init__()

        self.output_dim = output_dim
        self.batch_size = batch_size


        self.conv1 = nn.Conv2d(self.output_dim, dis_channel[0], stride = 2, kernel_size = (5,5), padding = 2)
        self.norm1 = nn.BatchNorm2d(dis_channel[0])

        self.conv2 = nn.Conv2d(dis_channel[0], dis_channel[1],  stride = 2, kernel_size = (5,5), padding = 2)
        self.norm2 = nn.BatchNorm2d(dis_channel[1])

        self.conv3 = nn.Conv2d(dis_channel[1], dis_channel[2],  stride = 2, kernel_size = (5,5), padding = 2)
        self.norm3 = nn.BatchNorm2d(dis_channel[2])

        self.conv4 = nn.Conv2d(dis_channel[2], dis_channel[3],  stride = 2, kernel_size = (5,5), padding = 2)
        self.norm4 = nn.BatchNorm2d(dis_channel[3])

        self.fc1 = nn.Linear(dis_channel[3] * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def leaky_relu(self, x, leak=0.2):
        return torch.max(x, leak * x)

    def forward(self, img):

        # Shape - Image : (4 x 128 x 128) (The channel dimension is 4 for PNG Image types and 3 for JPG image types)
        x = self.conv1(img)
        x = self.norm1(x)
        x = self.leaky_relu(x)

        # Shape - 32 x 64 x 64
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.leaky_relu(x)

        # Shape - 64 x 32 x 32
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.leaky_relu(x)

        # Shape - 128 x 16 x 16
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.leaky_relu(x)

        # Shape - 256 x 8 x 8
        x = x.view(self.batch_size,-1)
        # Shape - (batch_size x (256*8*8))
        x = self.fc1(x)
        x = self.leaky_relu(x)

        # Shape - (batch_size x 512)
        x = F.dropout(x, 0.5)

        x = self.fc2(x)
        # Shape - batch x 1
        return x
