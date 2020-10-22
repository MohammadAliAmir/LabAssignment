from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import random
import time
import math






def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )




# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]



# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self,  input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.num_layers = 1
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        out, hidden = self.lstm(input,hidden)
        out = self.linear(out)
        output = self.softmax(out)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1,1, self.hidden_dim)
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.batch_size=57
        self.hidden_size = hidden_size
        self.gru = nn.LSTM(input_size, hidden_size, 1)
        # self.i2h = nn.Linear((hidden_size,hidden_size), hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        # input = torch.unsqueeze(input,0)
        # input = input.transpose(1,0)
        # print(len(input))
        # hidden = torch.unsqueeze(hidden,0)
        lstm_out, hidden = self.gru(input,hidden)
        # output, hidden = self.gru(input)  # Hier moet ik input (Tensor) , tuple (Tensor, Tensor)
        # combined = torch.cat((lstm_out, hidden), 1)
        # hidden = self.i2h(hidden)
        output = self.h2o(hidden[0].view(1, self.hidden_size))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size)
    def init_hidden(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        return [t for t in (h0, c0)]


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden(line_tensor)
    # hidden.unsqueeze(0)
    rnn.zero_grad()

    # for i in range(line_tensor.size()[0]):
    #     output,hidden = rnn(line_tensor[i], hidden)
    output, hidden = rnn(line_tensor, hidden)
    # output=output.squeeze()
    # category_tensor=category_tensor.expand(output.size()[0])
    # for i in range(output.size()[0]):
    #     loss = criterion(output[i], category_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden(line_tensor)

    # for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor, hidden)

    return output


if __name__ == "__main__":
    print(findFiles('data/names/*.txt'))

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    print(unicodeToAscii('Ślusàrski'))
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(category_lines['Italian'][:5])
    print(letterToTensor('J'))

    print(lineToTensor('Jones').size())
    n_hidden = 256
    rnn = RNN(n_letters, n_hidden,n_categories)
    # 57 letters = input size, 128 = hidden size, 18 categories= output size
    # input = letterToTensor('A')
    # print(input)
    # hidden = torch.zeros(1,n_letters, n_hidden)
    # print(hidden)
    # input =input.unsqueeze(0)
    # # output = rnn(input)
    #
    # output, next_hidden = rnn(input, hidden)
    # input = lineToTensor('Albert')
    # hidden = torch.zeros(1, n_hidden)
    #
    # output, next_hidden = rnn(input, hidden)
    # print(output)
    # input = lineToTensor('Albert')
    # hidden = torch.zeros(1, n_hidden)
    #
    # output, next_hidden = rnn(input, hidden)
    # print(output)
    # print(categoryFromOutput(output))
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


    plt.figure()
    plt.plot(all_losses)
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()




