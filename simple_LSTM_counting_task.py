from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import torch.nn as nn
import torch
import argparse
import matplotlib.pyplot as plt
import os
import json

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 21, 20, 1
epochs = 300000000
seq_length = 100
data_time_steps = np.linspace(0, 99, 100)
criterion = nn.BCELoss()

# Things I want to save are a copy of the code the model and the training data plot.


def save_to_np(tensors,filename):
    np.savez(filename, *tensors)
    



class LSTMCell:

    def __init__(self, input_size, hidden_size):
        self.input_size=input_size
        self.W_full = torch.FloatTensor(input_size + hidden_size, hidden_size*4).type(dtype).to(device)
        init.normal(self.W_full, 0.0, 0.4)
        self.W_full = Variable(self.W_full, requires_grad = True)
        self.bias=1.0
        self.forget_bias=1.0

    def __call__(self, x, h, c):

        concat = torch.cat((x, h), dim=1)
        hidden = torch.matmul(concat, self.W_full)+self.bias

        i, g, f, o = torch.chunk(hidden, 4, dim=1)

        i = torch.sigmoid(i)
        g = torch.tanh(g)
        f = torch.sigmoid(f+self.forget_bias)
        o = torch.sigmoid(o)

        new_c = torch.mul(c, f) + torch.mul(g, i)
        new_h = torch.mul(torch.tanh(new_c), o)

        return new_h, new_c
   
    
    
    
class Model:
    def __init__(self, input_size, hidden_size, output_size, number_layers = 1):
        self.lstms = []
        self.lstms.append(LSTMCell(input_size,hidden_size))
        for lyr in range(1, number_layers):
            self.lstms.append(LSTMCell(hidden_size,hidden_size))
            
        self.W_final = torch.FloatTensor(hidden_size, 
        output_size).type(dtype).to(device)
        init.normal(self.W_final, 0.0, 0.4)
        self.W_final = Variable(self.W_final, requires_grad = True)
        
        self.number_layers = number_layers
        
    def __call__(self, x, hs, cs):
        
        for i in range(len(self.lstms)):
            h, c = self.lstms[i](x, hs[i], cs[i])
            hs[i], cs[i] = h, c
            x = h # at each level the h from h_n-1 becomes x_n
        
        y_ = torch.matmul(h, self.W_final)
        y_ = torch.sigmoid(y_)
        
        return y_, h, c
    
    def save(self, epoch):
        to_save = [parameter.detach().cpu().numpy() for parameter in self.parameters()]
        save_to_np(to_save, args.log_dir + "/models/model_epoch_" + str(epoch))
        
    def load(self, filename):
        data = np.load(filename)
        self.lstm.W_full.data, self.W_final.data = torch.from_numpy(data['arr_0']).to(device), torch.from_numpy(data['arr_1']).to(device)
        
    def parameters(self):
        
        parameters = []
        parameters.append(self.W_final)
        
        for lstm in self.lstms:
            parameters.append(lstm.W_full)
            
        return parameters
    
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size=input_size
        self.W_full = torch.FloatTensor(input_size + hidden_size, hidden_size).type(dtype).to(device)
        init.normal(self.W_full, 0.0, 0.4)
        self.W_full = Variable(self.W_full, requires_grad = True)
        self.bias=1.0
        self.forget_bias=1.0
        
    def __call__(self, x, h):
        concat = torch.cat((x, h), dim=1)
        hidden = torch.matmul(concat, self.W_full)+self.bias
        return torch.tanh(hidden)

    
class Model_RNN:
    def __init__(self, input_size, hidden_size, output_size, number_layers = 1):
        self.rnns = []
        self.rnns.append(RNNCell(input_size,hidden_size))
        for lyr in range(1, number_layers):
            self.rnns.append(RNNCell(hidden_size,hidden_size))
            
        self.W_final = torch.FloatTensor(hidden_size, 
        output_size).type(dtype).to(device)
        init.normal(self.W_final, 0.0, 0.4)
        self.W_final = Variable(self.W_final, requires_grad = True)
        
        self.number_layers = number_layers
        
    def __call__(self, x, hs):       
        for i in range(len(self.rnns)):
            h = self.rnns[i](x, hs[i])
            hs[i] =  h
            x = h # at each level the h from h_n-1 becomes x_n
        
        y_ = torch.matmul(h, self.W_final)
        y_ = torch.sigmoid(y_)
        
        return y_, h
    
    def save(self, epoch):
        to_save = [parameter.detach().cpu().numpy() for parameter in self.parameters()]
        save_to_np(to_save, args.log_dir + "/models/model_epoch_" + str(epoch))
        
    def load(self, filename):
        data = np.load(filename)
        self.rnn.W_full.data, self.W_final.data = torch.from_numpy(data['arr_0']).to(device), torch.from_numpy(data['arr_1']).to(device)
        
    def parameters(self):
        
        parameters = []
        parameters.append(self.W_final)
        
        for rnn in self.rnns:
            parameters.append(rnn.W_full)
            
        return parameters
    
        
def count_task(seq_len, seed):
    
    np.random.seed(seed)
    labels = np.zeros([1,seq_len])
    X =  np.random.randint(2, size=[1,seq_len])
    #X[0,:] = 1.0
    len_streak = 0
    max_streak = 0
    for ii in range(seq_len):
        if X[0,ii] == 1:
            len_streak += 1
            max_streak = max(max_streak, len_streak)
# simplified the task now it is just counting how many 1
        else:    
            len_streak = 0
            
    labels[0, :max_streak] = 1
    dummy_X = np.zeros([1, seq_len])
    dummy_X[0] = 1 # mark the beginning of the evaluation period
    return np.concatenate([X, dummy_X], axis=1), np.concatenate([np.zeros([1, seq_len]), labels], axis=1)
        
    
        
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--log-dir', default='', help='directory where you want to save the data')
    parser.add_argument(
        '--device', default='', help='cuda device')
    parser.add_argument(
        '--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument(
        '--mod',type=int, default=5, help='modulo for generating the training data ')
    parser.add_argument(
        '--seq-len',type=int, default=10, help='legth of the training sequence')
    parser.add_argument(
        '--restart-model', default=None, help='restart model file')
    parser.add_argument(
        '--cl', action='store_true',default=None ,help=' The agent starts with low seq number and increases incrementally ')
    parser.add_argument(
        '--n-layers',type=int, default=1, help='number of layers of the lstm')
    parser.add_argument(
        '--lstm', action='store_true',default=False ,help=' LSTM ')
    

    args = parser.parse_args()

    return args

args = get_args()
args_dict = vars(args)

os.system("mkdir "+ args.log_dir)
json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)
os.system("cp simple_LSTM.py "+ args.log_dir + "/simple_LSTM_copy.py")
os.system("mkdir "+ args.log_dir + "/models")

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print("DEVICE", device)

if args.lstm:
    model = Model(1, 20, 1, number_layers=args.n_layers)
else:
    model = Model_RNN(1, 20, 1, number_layers=args.n_layers)

if args.restart_model:
    model.load(args.restart_model)
    
batch_size = 10

# lr suggested = 0.00001 
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr )

if args.cl:
    seq_len = 2
else:
    seq_len = args.seq_len

losses = []

for i in range(epochs):
    optimizer.zero_grad()
    total_loss = 0
    
    for kk in range(batch_size):
        hs = []
        cs = []
        seed = np.random.randint(200)
        X, Y = count_task(10, seed)

        X = Variable(torch.Tensor(X).type(dtype), requires_grad=False).squeeze(0)
        Y = Variable(torch.Tensor(Y).type(dtype), requires_grad=False).squeeze(0)

        for kk in range(model.number_layers):
            c = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad = True).to(device)
            h = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad = True).to(device)
            hs.append(h)
            cs.append(c)

    #     X_small = X[:seq_len]
    #     Y_small = Y[:seq_len]

        y_s = np.array([0 for i in range(len(Y))])
        for j in range(X.size(0)):
            x = X[j].unsqueeze(0).unsqueeze(0).to(device)
            y = Y[j].unsqueeze(0).unsqueeze(0).to(device)
            if args.lstm:
                y_, h, c = model(x,hs,cs)
            else:
                y_, h = model(x,hs)
            y_s[j] = y_
#             if i % 100 == 0:
#                 print(y_)

            loss = criterion( y_, y)

            total_loss += loss

    #         model.W_full.grad.data.zero_()
    #         model.W_final.grad.data.zero_() 
            for kk in range(model.number_layers):
                if args.lstm:
                    c = cs[kk]
                h = hs[kk]
                if args.lstm:
                    c = Variable(c.data, requires_grad = True)
                    
                h = Variable(h.data, requires_grad = True)
            #print(context_state.requires_grad)
    #     if seq_len < X.size(0) and i != 0 and i % 20000 == 0 and args.cl:
    #         seq_len += 2
    #         print("-------", seq_len)

    
    total_loss.backward()    
    optimizer.step()


    
    
    if i % 100 == 0:
        print("Epoch: {} loss {}".format(i, total_loss.item()/batch_size))
    if i % 1 == 0:
        losses.append(total_loss.item()/batch_size)
        
        
    if i % 100 == 0:
        plt.plot(losses)
        plt.savefig(args.log_dir + '/loss_plot.png')
        plt.close()

#         Y2 = np.array([1.0 if i % args.mod == 0 else 0.0 for i in range(args.seq_len)] )
#         X2 = np.array([0 for i in range(args.seq_len)] )
        
#         plt.plot(X2[:seq_len], Y2[:seq_len])
#         plt.plot(X2[:seq_len], y_s)
#         plt.savefig(args.log_dir + '/test_plot.png')
#         plt.close()
        
    
        
    if i % 20000 == 0:
        model.save(i)
        


