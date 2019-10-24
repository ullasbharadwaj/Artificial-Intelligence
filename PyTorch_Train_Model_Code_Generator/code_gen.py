import numpy as np
import argparse
import ast
import sys

"""
Version : 0.01

Example Usage CNN:
python code_gen.py --file_name 'Model_Autogen' --network 'CNN' --num_of_conv_FF_layers '[No_ConvLayers, No_DenseLayers]' --CNN_Hyperparameters '[Input_Dim, Filters_layer_1, Filters_layer_2,...., Out_Dim, Dense1, Dense2, Dense_Out, Batch_Normalization(0/1)]'
python code_gen.py --file_name 'Model_Autogen' --network 'CNN' --num_of_conv_FF_layers '[3,2]' --CNN_Hyperparameters '[4,16,32,64,16,512,512,37,1]'

Example Usage FF:
python code_gen.py --file_name 'Model_Autogen' --network 'FF' --FF_Hyperparameters '[Number_Hidden_Layers, Input_Dim, Dense1, Dense2,...., Out_Dim]'
python code_gen.py --file_name 'Model_Autogen' --network 'FF' --FF_Hyperparameters '[3,16,512,512,10]'

"""
def Generate_CNN_Skeleton(filename, num_conv_Layers, num_FF_Layers, Filters, Dense_Neurons, batch_norm):

    f.write('##### Class defining the CNN Model ######\n')

    f.write('class AutoGen_CNN(nn.Module):\n')

    f.write('\tdef __init__(self, TODO:Input Dimension):\n')

    f.write('\t\tsuper(AutoGen_CNN,self).__init__()\n\n')

    f.write('######### Model Definition #########\n')

    for conv_layer in range(1,num_conv_Layers+1):
        f.write('\t\tself.conv'+str(conv_layer)+' = nn.Conv2d('+str(Filters[conv_layer])+','+ str(Filters[conv_layer+1]) + ', stride = TODO: Choose Stride, kernel_size = TODO: (f,f), padding = TODO: (p,p))\n')
        if batch_norm == 1:
            f.write('\t\tself.norm'+str(conv_layer)+' = nn.BatchNorm2d('+ str(Filters[conv_layer+1])+ ')\n\n')

    f.write('\t\tself.ff'+str(1)+' = nn.Linear(TODO:InputDimension, '+str(Dense_Neurons[0])+')\n\n')
    for FF_layer in range(num_FF_Layers-1):
        f.write('\t\tself.ff'+str(FF_layer+2)+' = nn.Linear('+str(Dense_Neurons[FF_layer])+','+str(Dense_Neurons[FF_layer+1])+')\n\n')

    f.write('\t\tself.device = "cpu"\n')

    f.write('\t\tif torch.cuda.is_available():\n\t\t\tself.device = "cuda"\n\n')

    f.write('######### Forward Pass #########\n')

    f.write('\tdef forward(self, input):\n\n')

    f.write('\t\tx = self.conv'+str(1)+'(input)\n')
    if batch_norm == 1:
        f.write('\t\tx = self.norm'+str(1)+'(x)\n')
    f.write('\t\tx = F.relu(x)\n\n')

    for conv_layer in range(2,num_conv_Layers+1):
        f.write('\t\tx = self.conv'+str(conv_layer)+'(x)\n')
        if batch_norm == 1:
            f.write('\t\tx = self.norm'+str(conv_layer)+'(x)\n')
        f.write('\t\tx = F.relu(x)\n\n')

    for FF_layer in range(1, num_FF_Layers+1):
        f.write('\t\tx = self.ff'+str(FF_layer)+'(x)\n')
        if FF_layer < num_FF_Layers:
            f.write('\t\tx = F.relu(x)\n\n')

    f.write('\t\treturn F.sigmoid(x)')


def Generate_FF_Skeleton(filename, num_hidden_layers, Dense_Neurons):

    f.write('##### Class defining the FF Model ######\n')

    f.write('class AutoGen_FF(nn.Module):\n')
    f.write('\tdef __init__(self, TODO:Input Dimension):\n')
    f.write('\t\tsuper(AutoGen_FF,self).__init__()\n\n')

    f.write('######### Model Definition #########\n')

    for FF_layer in range(num_hidden_layers):
        f.write('\t\tself.ff'+str(FF_layer+1)+' = nn.Linear(' + str(Dense_Neurons[FF_layer]) + ',' + str(Dense_Neurons[FF_layer+1])+')\n\n')

    f.write('\t\tself.device = "cpu"\n')
    f.write('\t\tif torch.cuda.is_available():\n\t\t\tself.device = "cuda"\n\n')

    f.write('######### Forward Pass #########\n')
    f.write('\tdef forward(self, input):\n\n')

    f.write('\t\tx = self.ff'+str(1)+'(input)\n')
    f.write('\t\tx = F.relu(x)\n\n')

    for FF_layer in range(2,num_hidden_layers+1):
        f.write('\t\tx = self.ff'+str(FF_layer)+'(x)\n')
        if FF_layer < num_hidden_layers:
            f.write('\t\tx = F.relu(x)\n\n')


    f.write('\n\t\treturn F.sigmoid(x)')



def Gen_Dataloader():

    f.write('##### Class defining the Dataloader ######\n')

    f.write('class DNN_Dataloader(Dataset):\n\n')
    f.write('\tdef __init__(self, data_folder_path):\n')

    f.write('\t\tself.data_files = glob.glob(data_folder_path + "/*.TODO:File Extension")\n')
    f.write('\t\tself.train_len = len(self.data_files)\n')
    f.write('\t\tself.data_folder_path = data_folder_path \n')
    f.write('\t\tself.trans= Compose([ToTensor()])\n\n')

    f.write('\tdef __len__(self):\n')
    f.write('\t\treturn self.train_len\n\n')

    f.write('\tdef __getitem__(self,idx):\n')
    f.write('\t\twith open(self.data_files[idx], "rb" ) as handle:\n')
    f.write('\t\t\t#TODO:Load the data file, Ex: Pickle Load/Image Load\n')
    f.write('\t\treturn self.trans(sample)\n\n')

def Gen_Train_Code(network_type):

    f.write('train_dataset = DNN_Dataloader(TODO:Give the train data folder path)\n')
    f.write('train_dataloader = DataLoader(train_dataset, batch_size = TODO, shuffle = True, drop_last = True )\n\n')

    f.write('validation_dataset = Dataloader(TODO:Give the validation data folder path)\n')
    f.write('validation_dataloader = DataLoader(validation_dataset, batch_size = TODO, shuffle = True, drop_last = True )\n\n')

    f.write('DNN_Model = AutoGen_'+network_type+'(TODO:Any arguments necessary)\n\n')

    f.write('DNN_Model.train()\n\n')
    f.write('DNN_Model.to(device)\n\n')

    f.write('optimizer = TODO: Define an optimizer\n\n')

    f.write('############################ Start Training ################################\n\n')

    f.write('for epoch in range(1, num_epochs+1)\n')
    f.write('\tfor batch in train_dataloader:\n')
    f.write('\t\toptimizer.zero_grad()\n\n')
    f.write('\t\t"""#################\n')
    f.write('\t\tDefine all the inputs to the forward pass and convert them to Torch tensors\n')
    f.write('\t\t#################"""\n\n')
    f.write('\t\tdnn_out = AutoGen_'+network_type+'(Fwd Pass Inputs)\n')
    f.write('\t\tbatch_loss = TODO:Define loss function\n')
    f.write('\t\tbatch_loss = batch_loss.mean()\n')
    f.write('\t\tbatch_loss.backward()\n')
    f.write('\t\toptimizer.step()\n\n\n')

    f.write('############################ Start Validating ################################\n\n')

    f.write('\tfor batch in validation_dataloader:\n')
    f.write('\t\twith torch.no_grad()\n\n')
    f.write('\t\t\t"""#################\n')
    f.write('\t\t\tDefine all the inputs to the forward pass and convert them to Torch tensors\n')
    f.write('\t\t\t#################"""\n\n')

    f.write('\t\t\tdnn_out = AutoGen_'+network_type+'(Fwd Pass Inputs)\n')
    f.write('\t\t\tbatch_loss = TODO:Define loss function\n')
    f.write('\t\t\tbatch_loss = batch_loss.mean()\n')

    f.write('\t\t\t"""#################\n')
    f.write('\t\t\tAnalyze the error for early stopping and save the model when necessary\n')
    f.write('\t\t\t#################"""\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code_Gen')

    parser.add_argument(
        '--file_name',
        help='name of the file',
        type=str,
        default='model'
    )

    parser.add_argument(
        '--network',
        help='type of network',
        type=str,
        default='FF'
    )
    ############## CNN related details ###############
    parser.add_argument(
        '--num_of_conv_FF_layers',
        help='Number of Conv Layers',
        type=str,
        default='[3,2]'
    )
    #[Input_Dim, Filters_layer_1,Filters_layer_2,....,Out_Dim, Dense1, Dense2, Batch_Normalization(0/1)]
    parser.add_argument(
        '--CNN_Hyperparameters',
        help='hyperparameters',
        type=str,
        default='[3,16,32,64,1]'
    )
    ############## FF related details ###############
    #[Number_Hidden_Layers, Input_Dim, Dense1,Dense2,....,Out_Dim]
    parser.add_argument(
        '--FF_Hyperparameters',
        help='hyperparameters',
        type=str,
        default='[3,16, 512, 512, 10]'
    )

    ############## LSTM related details ###############
    parser.add_argument(
        '--num_of_lstm_FF_layers',
        help='Number of LSTM and FF Layers',
        type=str,
        default='[2,1]'
    )


    args = parser.parse_args()
    filename = args.file_name
    network_type = args.network
    with open(filename+'.py', 'w+') as f:

        f.write('"""\n\tThis is an auto-generated Python Script. \n\tFill in the TODO blanks as needed.\n\n"""\n')
        f.write('######### Importing all necessary packages #########\n')
        f.write('import torch \nimport torch.nn as nn \nimport torch.nn.functional as F \nimport torch.optim as optim\n\n')
        f.write('from torch.utils.data import DataLoader\nfrom torchvision import datasets\nfrom torchvision.transforms import Compose,ToTensor\n\n')

        if network_type == 'CNN':

            num_of_conv_FF_layers = args.num_of_conv_FF_layers
            num_of_conv_FF_layers = ast.literal_eval(num_of_conv_FF_layers)
            Conv_Hyperparameters = args.CNN_Hyperparameters
            Conv_Hyperparameters = ast.literal_eval(Conv_Hyperparameters)

            num_conv_Layers = num_of_conv_FF_layers[0]
            num_FF_Layers = num_of_conv_FF_layers[1]
            Filters = Conv_Hyperparameters[:num_conv_Layers+2]
            Dense_Neurons = Conv_Hyperparameters[num_conv_Layers+2:-1]
            batch_norm = Conv_Hyperparameters[-1]
            Generate_CNN_Skeleton(filename, num_conv_Layers, num_FF_Layers, Filters, Dense_Neurons, batch_norm)

        if network_type == 'FF':

            FF_Hyperparameters = args.FF_Hyperparameters
            FF_Hyperparameters = ast.literal_eval(FF_Hyperparameters)

            num_hidden_layers = FF_Hyperparameters[0]
            Dense_Neurons = FF_Hyperparameters[1:]
            Generate_FF_Skeleton(filename, num_hidden_layers, Dense_Neurons)

        f.write('\n\n')
        Gen_Dataloader()
        f.write('\n\n')
        f.write('self.device = "cpu"\n')
        f.write('if torch.cuda.is_available():\n')
        f.write('\tdevice = "cuda"\n\n')
        f.write('\tnum_epochs = TODO\n\n')
        Gen_Train_Code(network_type)
        f.close()
