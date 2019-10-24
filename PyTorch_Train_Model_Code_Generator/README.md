Code Generation for training DNNs:

This script generates a skeleton of Python source file with Model description, Training module and the Dataloader based on the 
parameter information given by the user which can then be used for training.  

Placeholders for vital operations are indicated with TODO texts which needs to be taken care and tailored befor it can be used for training.  

Current version 0.01 supports code generation for Convolutional Neural Networks and Feed Forward DNNs.

Example Usage CNN:
python code_gen.py --file_name 'Model_Autogen' --network 'CNN' --num_of_conv_FF_layers '[No_ConvLayers, No_DenseLayers]' --CNN_Hyperparameters '[Input_Dim, Filters_layer_1, Filters_layer_2,...., Out_Dim, Dense1, Dense2, Dense_Out, Batch_Normalization(0/1)]'

python code_gen.py --file_name 'Model_Autogen' --network 'CNN' --num_of_conv_FF_layers '[3,2]' --CNN_Hyperparameters '[4,16,32,64,16,512,512,37,1]'

Example Usage FF:
python code_gen.py --file_name 'Model_Autogen' --network 'FF' --FF_Hyperparameters '[Number_Hidden_Layers, Input_Dim, Dense1, Dense2,...., Out_Dim]'

python code_gen.py --file_name 'Model_Autogen' --network 'FF' --FF_Hyperparameters '[3,16,512,512,10]'
