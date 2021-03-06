import numpy as np
import matplotlib.pyplot as plt
#import cma
from es import SimpleGA, CMAES, PEPG, OpenES

def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

def BCE_loss(y,p):
    return np.sum(-y*np.log(p + 0.0000000001) - (1 - y)*np.log(1 - p + 0.0000000001))


def binarize(x):
    res = x > 0.5
    return res.astype(int)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def passthru(x):
  return x

# useful for discrete actions
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

# useful for discrete actions
def sample(p):
  return np.argmax(np.random.multinomial(1, p))


class RNNCell:
  def __init__(self, input_size, weight, bias):
    self.input_size=input_size
    self.weight = weight
    self.bias = bias
  def __call__(self, x, h):
    concat = np.concatenate((x, h), axis=1)
    hidden = np.matmul(concat, self.weight)+self.bias
    return np.tanh(hidden)

class RNNModel:
    def __init__(self):
    

        self.hidden_size = 30

        self.layer_1 = 30
        self.layer_2 = 30

        self.rnn_mode = True

        self.input_size = 1
        self.output_size = 1
        self.alpha = 45


        self.shapes = [ (self.input_size + self.hidden_size, 1*self.hidden_size), # RNN weights
                        (self.input_size + self.hidden_size, self.layer_1),# predict actions output
                        (self.layer_1, self.output_size)] # predict actions output

        self.weight = []
        self.bias = []
        self.param_count = 0

        idx = 0
        for shape in self.shapes:
          self.weight.append(np.zeros(shape=shape))
          self.bias.append(np.zeros(shape=shape[1]))
          self.param_count += (np.product(shape) + shape[1])
          idx += 1

        self.init_h = np.zeros((1, self.hidden_size))
        self.h = self.init_h
        self.param_count += 1*self.hidden_size

        self.rnn = RNNCell(self.input_size, self.weight[0], self.bias[0])

    def reset(self):
        
       
        self.count = 0
        if   self.alpha <= self.count:
            self.h = binarize(sigmoid(self.init_h))
        else:
             self.h =  sigmoid(self.init_h)
        


    def get_action(self, x):
        obs = x.reshape(1, self.input_size)
        self.count += 1

        # update rnn:
        #update_obs = np.concatenate([obs, action], axis=1)
         
#         if  self.alpha <= self.count:
        self.h = binarize(sigmoid(self.rnn(x, self.h)))
        x = np.concatenate([x, self.h], axis=1)

#         else:
#             self.h = sigmoid(self.rnn(x, self.h))
#             x = np.concatenate([x, self.h], axis=1)

        # calculate action using 2 layer network from output
        hidden = np.tanh(np.matmul(x, self.weight[1]) + self.bias[1])
        action = sigmoid(np.matmul(hidden, self.weight[2]) + self.bias[2])

        return action[0]

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
          w_shape = self.shapes[i]
          b_shape = self.shapes[i][1]
          s_w = np.product(w_shape)
          s = s_w + b_shape
          chunk = np.array(model_params[pointer:pointer+s])
          self.weight[i] = chunk[:s_w].reshape(w_shape)
          self.bias[i] = chunk[s_w:].reshape(b_shape)
          pointer += s
        # rnn states
        s = self.hidden_size
        self.init_h = model_params[pointer:pointer+s].reshape((1, self.hidden_size))
        self.h = self.init_h
        self.rnn = RNNCell(self.input_size, self.weight[0], self.bias[0])

    def load_model(self, filename):
        with open(filename) as f:    
          data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0]) # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count)*stdev

    def update_alpha(self):
        self.alpha = self.alpha -1


model = RNNModel()

#model.get_action(np.array([[4]]))


NPARAMS = model.param_count   # make this a 100-dimensinal problem.
NPOPULATION = 450    # use population size of 101.
MAX_ITERATION = 4010 # run each solver for 5000 generations.


def recurrency_label(seq_len):
    
    labels = []
    X = []

    X = np.zeros([seq_len,1])
    #X[0,:] = 1.0
    for ii in range(seq_len):
        if ii % 61 == 0:
            labels.append(np.ones([1,1]))

        else:
            labels.append(np.zeros([1,1]))

    return X, np.concatenate(labels, axis=0)



def evluate_func(data):
    model, params =  data
    model.set_model_params(params)
    model.reset()
    loss_cum = 0
    Xs, labels  = recurrency_label(64)

    for x, label in zip(Xs,labels):
        

        x = np.array([x])
        pred = model.get_action(x)
        
        #print(label, pred)
#         if model.count >= model.alpha:
        loss = np.abs(BCE_loss(label, pred))
        loss_cum += loss
    #print(loss_cum)
    return -loss_cum

def evluate_func_test(model, params):
#     model, target_model, params =  data
    model.set_model_params(params)
    model.reset()

    loss_cum = 0
    Xs, labels  = recurrency_label(63)

    for x, label in zip(Xs,labels):
        
        x = np.array([x])
        
        pred = model.get_action(x)
        
        print(model.h)
        
        loss = np.abs(BCE_loss(label, pred))
        print(label, pred, loss)
        

        loss_cum += loss
    #print(loss_cum)
    return -loss_cum

# defines genetic algorithm solver
ga = SimpleGA(NPARAMS,                # number of model parameters
               sigma_init=0.5,        # initial standard deviation
               popsize=NPOPULATION,   # population size
               elite_ratio=0.1,       # percentage of the elites
               forget_best=False,     # forget the historical best elites
               weight_decay=0.00,     # weight decay coefficient
              )

import multiprocessing as mp

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
fit_func = evluate_func
# defines a function to use solver to solve fit_func
def test_solver(solver):
    history = []
    j = 0
    while True:
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        #print(solutions)
        fitness_list = pool.map(fit_func, [(model,solutions[i]) for i in range(solver.popsize)])
            

        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j+1) % 10 == 0:
            print("fitness at iteration", (j+1), result[1])
            print("Best:",solver.elite_rewards[0])
            
#             if -solver.elite_rewards[0] < 0.00001:
#                 model.update_alpha()
#                 print('Alpha changed to', model.alpha)
#                 model.first_iteration = True
#                 new_elite_rewards = []
#                 new_elite_params = solver.elite_params
#                 for kk in range(len(solver.elite_rewards)):
# #                     print('old',solver.elite_rewards[kk])
#                     new_elite_rewards.append(fit_func((model,solver.elite_params[kk])))
    
#                 solver = SimpleGA(NPARAMS,                # number of model parameters
#                        sigma_init=0.5,        # initial standard deviation
#                        popsize=NPOPULATION,   # population size
#                        elite_ratio=0.05,       # percentage of the elites
#                        forget_best=False,     # forget the historical best elites
#                        weight_decay=0.00,     # weight decay coefficient
#                       )
#                 solver.elite_params = new_elite_params
#                 solver.elite_rewards = new_elite_rewards
# #                     print('new',solver.elite_rewards[kk])
# #                     print('new_batch', solver.elite_rewards[0:5])

        if (j+1) % 100 == 0:   
            evluate_func_test(model, result[0])

        if -result[1] <= 0.0000001 and model.alpha <= 0:
            print("local optimum discovered by solver:\n", result[0])
            print("fitness score at this local optimum:", result[1])
            return history, result
        j += 1
        
             
    

ga_history, result = test_solver(ga)
evluate_func_test(model, result[0])
