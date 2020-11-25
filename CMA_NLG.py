import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES
import torch
import torch.nn as nn 


def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

# def BCE_loss(y,p):
#     return np.sum(-y*np.log(p) - (1 - y)*np.log(1 - p))

criterion = nn.BCELoss()

class gate_layer():
    def __init__(self,dims, n_channels= 1, AND_OR = 1):
        
        # AND_OR = TRUE  : it is an AND gate
        # AND_OR = FALSE : it is an OR gate
        
        self.AND_OR = AND_OR
        self.n_channels = n_channels
        dims[0] = n_channels
        self.w =  np.random.normal(-3, 1, dims)
  
    def forward(self, X):
       
        X = np.repeat(X, self.n_channels, axis=0)

        m = sigmoid(self.w)

        if self.AND_OR == 1:
            filtered_X = 1 - m *(1 - X) 
            
            return np.prod(filtered_X,1)[np.newaxis,...]
        
        if self.AND_OR == 2:

            filtered_X = m*X
           
            return (1- np.prod(1 - filtered_X,1))[np.newaxis,...]
        
        if self.AND_OR == 3:
            # inverter
            filtered_X = m*X
            res = ones - filtered_X

            return res
            
    
class Model():
    
    def __init__(self):
        
        self.layers = []
        self.lyr_shapes = []

        
    def add_layer(self,dims, n_channels, AND_OR=True):
        # add output size arg
        
        lyr = gate_layer(dims, n_channels=n_channels, AND_OR=AND_OR)
        self.layers.append(lyr)
        dims[0] = n_channels
        self.lyr_shapes.append(np.array(dims))

        
    def forward(self,x):
        
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def params(self):
        # returns a list of the parameters to give to the optimizer
        self.para = []
        for layer in self.layers:       
            self.para.append(layer.w)     
        return self.para
    
    def set_model_params(self, model_params):
        s1 = 0
        for layer, lyr_shape in zip(self.layers,self.lyr_shapes ):
            
            s2 = np.prod(lyr_shape)
            layer.w = np.copy(model_params[s1:s1+s2].reshape(lyr_shape))
            s1 += s2
            
    def num_params(self):
        tot_num = 0
        for lyr_shape in self.lyr_shapes:
            tot_num += np.prod(lyr_shape)
            
        return tot_num
            
        
    
def recurrency_label(seq_len):
    
    labels = []
    X = []

    X = np.zeros([seq_len,1])
    #X[0,:] = 1.0
    for ii in range(seq_len):
        
        if ii % 7 == 0:
            labels.append(np.ones([1,1]))

        else:
            labels.append(np.zeros([1,1]))

    
    return X, np.concatenate(labels, axis=0)



def evluate_func(model, params):
    model.set_model_params(params)
    loss_cum = 0
    Xs, labels  = recurrency_label(10)
    
#     X = np.ones([1,1])
#     inv_X = 1 - X

#     aug_X = np.concatenate([X, inv_X], axis=1)
    h_ = np.zeros([1,10])
    h_[0,0] = 1.0
#     xh = np.concatenate([aug_X,h_], axis=1)
#     _, h_ = model.forward(xh)

    for X, label in zip(Xs,labels):
        
        inv_X = 1- X
        
        aug_X = np.concatenate([X, inv_X], axis=0)
        
        xh = np.concatenate([aug_X[np.newaxis,...],h_], axis=1)

        out= model.forward(xh)
        
        pred = out[:,0:1]
        h_ = np.copy(out[:,1:])

        #print(label, pred)
        pred, label =  torch.from_numpy(pred), torch.from_numpy(label[0])
        loss = criterion(pred, label ).numpy()
        
        loss_cum += loss
    
    return loss_cum


model = Model()
model.add_layer([1,12], n_channels = 20, AND_OR=1)
model.add_layer([1,20], n_channels = 11, AND_OR=2)



NPARAMS = model.num_params()       # make this a 100-dimensinal problem.
NPOPULATION = 4000    # use population size of 101.
MAX_ITERATION = 4000 # run each solver for 5000 generations.



cmaes = CMAES(NPARAMS,
              popsize=NPOPULATION,
              weight_decay=0.0,
              sigma_init = 0.5
          )


fit_func = evluate_func
# defines a function to use solver to solve fit_func
def test_solver(solver):
    history = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        #print(solutions)
        for i in range(solver.popsize):
            fitness_list[i] = -fit_func(model,solutions[i])
            #print(fit_func(model,solutions[i]))
            

        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j+1) % 10 == 0:
            print("fitness at iteration", (j+1), result[1])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history

cma_history = test_solver(cmaes)
