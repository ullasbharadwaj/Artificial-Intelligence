"""
######################################################################
            *** Source code for performing Text Generation ***
                            using RNNs.
######################################################################
"""

"""
Import all the required packages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
from argparse import Namespace
import ipdb
import glob

"""
Initialize all the required hyper-parameters and other input information
"""
flags = Namespace(
    train_file='input.txt',
    seq_size=16,
    batch_size=8,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['One', 'should'],
    predict_top_k=6,
    checkpoint_path='checkpoint',
)

Number_of_Epochs = 1000

"""
Function to get input data, true label vector, number of words info from the Input file used for training
"""
def Get_Info_From_File(train_file, batch_size, seq_size):
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.split()
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    # Getting the integer and word representations from the input
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    number_vocab = len(int_to_vocab)

    # Converting the set of words in the input text to integer representations
    int_text = [vocab_to_int[w] for w in text]

    #Calculate the number of batches from the calculated number of words
    number_batches = int(len(int_text) / (seq_size * batch_size))

    #Generate the Input training data in the integer form
    X_text = int_text[:number_batches * batch_size * seq_size]

    #Generate the True Text Vector cooresponding to X_test
    Y_text = np.zeros_like(X_text)
    Y_text[:-1] = X_text[1:]
    Y_text[-1] = X_text[0]
    X_text = np.reshape(X_text, (batch_size, -1))
    Y_text = np.reshape(Y_text, (batch_size, -1))

    return int_to_vocab, vocab_to_int, number_vocab, X_text, Y_text


"""
Setting up the dataloader for training
"""
def Generate_Batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    # print('Number of Batches: {}'.format(num_batches))
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


"""
Class representing the DNN module.
It uses an LSTM FOLLOWED BY A SINGLE DENSE LAYER trained with word embeddings.
"""
class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, LSTM_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.LSTM_size = LSTM_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            LSTM_size,
                            batch_first=True)
        self.dense = nn.Linear(LSTM_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, hidden_state = self.lstm(embed, prev_state)
        dnn_out = self.dense(output)
        return dnn_out, hidden_state

    def init_hidden_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.LSTM_size),torch.zeros(1, batch_size, self.LSTM_size))


"""
Function to define the optimizer with suitable hyper-parameters.
We use Adam Optimzer with an initial learning rate of 0.001.
"""
def Get_Loss(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


"""
Function to run predictions using the trained network
"""
def predict(device, net, words, number_of_words , vocab_to_int, int_to_vocab, top_k=5):

    state_h, state_c = net.init_hidden_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(number_of_words):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(' '.join(words).encode('utf-8'))

"""
Test routine to generate the texts using the trained LSTM network. The 'Predict' subroutine is invoked from this function.
"""
def Test(int_to_vocab, vocab_to_int, n_vocab):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device= 'cpu'

    number_of_words = 50
    number_of_sentences = 50

    #Load the latest trained model in the directory storing the trained models
    list_of_models= glob.glob('./checkpoint/*.pth')
    latest_model = max(list_of_models, key=os.path.getctime)

    #Instantiate the RNN module
    net = RNNModule(n_vocab , flags.seq_size, flags.embedding_size, flags.lstm_size)
    #Load the state dictionary of the trained module
    net_trained = net.load_state_dict(torch.load(latest_model))
    #Transfer the model to GPU/CPU based on availability
    net = net.to(device)
    #Put the model in the evaluation mode
    net.eval()

    #Option to chose the starting two words of the sentence
    words = ['One', 'should']
    for i in range(number_of_sentences):
        predict(device, net, words, number_of_words, vocab_to_int, int_to_vocab, top_k=5)

"""
DNN Training sub-routine to train the LSTM Network
"""
def Train():
    print('Number of Epochs: {}'.format(Number_of_Epochs))

    # Set the device to GPU/CPU based on availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate the Input and Label Data Vectors from the input text file
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = Get_Info_From_File(
        flags.train_file, flags.batch_size, flags.seq_size)

    #Instantiate the RNN module
    net = RNNModule(n_vocab, flags.seq_size,flags.embedding_size, flags.lstm_size)

    #Transfer the model to GPU/CPU based on availability
    net = net.to(device)

    #Set up the Optimizer
    criterion, optimizer = Get_Loss(net, 0.01)

    iteration = 0
    old_loss = 100e2
    early_stop_flag = 0

    for epoch in range(Number_of_Epochs):
        #Set up the Training data in terms of batches
        batches = Generate_Batches(in_text, out_text, flags.batch_size, flags.seq_size)
        #Initialze the Hidden States and Cell States
        state_h, state_c = net.init_hidden_state(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        iteration = 0
        for x, y in batches:
            iteration += 1
            net.train()

            optimizer.zero_grad()

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            #Get the output from DNN. The return values are, Hidden State of the last layer (dnn_out), hidden states(state_h) and cell states(state_c)
            dnn_out, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(dnn_out.transpose(1, 2), y)

            loss_value = loss.item()

            #Check for early Stopping
            if (old_loss > loss_value) or e == 0:
                old_loss = loss_value
                old_dict = net.state_dict()
                early_stop_flag = 0
            else:
                early_stop_flag += 1
                old_loss = loss_value
                if early_stop_flag > 10:
                    torch.save(old_dict,'checkpoint/model-{}.pth'.format(iteration))
                    print('Stopping the training process....Exiting!!')
                    exit()

            #Propagate the Loss backward to adjust the DNN parameters
            loss.backward()

            state_h = state_h.detach()
            state_c = state_c.detach()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % 1 == 0:
                print('Epoch: {}/{}'.format(epoch, Number_of_Epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value),end='\r')

        #Run predictions every 5 epochs
        if epoch % 5 == 0:
            number_of_words = 10
            predict(device, net, flags.initial_words, number_of_words, vocab_to_int, int_to_vocab, top_k=5)
        #Save Models every 10 epochs
        if epoch % 10 == 0:
            torch.save(net.state_dict(),'checkpoint/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    train= True
    if train==True:
        Train()
    else:
        int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)
        Test(int_to_vocab, vocab_to_int, n_vocab)
