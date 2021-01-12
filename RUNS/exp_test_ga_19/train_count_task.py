import numpy as np
import matplotlib.pyplot as plt
#import cma
from es import SimpleGA, CMAES, PEPG, OpenES
import json
import argparse
import multiprocessing as mp
import os
import matplotlib.pyplot as plt

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
    def __init__(self, binary, multiscale):
    
        self.binary = binary
        self.hidden_size = 30
        self.multiscale = multiscale

        self.layer_1 = 30
        self.layer_2 = 30
        self.n_layers = 3

        self.rnn_mode = True

        self.input_size = 1
        self.output_size = 1

        self.freqs = []
        self.shapes  = [] 
        for lyr_n in range(self.n_layers):
            if lyr_n == 0:
                self.shapes.append((self.input_size + self.hidden_size, 1*self.hidden_size))
            else:
                self.shapes.append((2*self.hidden_size, 1*self.hidden_size))
                
            self.freqs.append(10**lyr_n)
            
         
        
        self.shapes +=  [ (self.input_size +  self.n_layers*self.hidden_size, self.layer_1),# predict actions output
                        (self.layer_1, self.output_size)]   # predict actions output
        
        
        
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
        self.hs = []
        #self.param_count += 1*self.hidden_size
        self.rnns = []
        for rnn_n in range(self.n_layers):
            if lyr_n == 0:
                self.rnns.append(RNNCell(self.input_size, self.weight[rnn_n], self.bias[rnn_n]))
            else:
                 self.rnns.append(RNNCell(self.hidden_size, self.weight[rnn_n], self.bias[rnn_n]))

    def reset(self):
        self.hs = []
        for kk in range(self.n_layers):
            if self.binary:
                self.hs.append(binarize(sigmoid(self.init_h)))
            else:
                self.hs.append(sigmoid(self.init_h))
        self.step = 0

        
    def get_action(self, x):
        x = x.reshape(1, self.input_size)
        #obs = x

        for jj in range(self.n_layers):
            if jj == 0:
                if self.binary:
                    self.hs[jj] = np.copy(binarize(sigmoid(self.rnns[jj](x, self.hs[jj]))))
                else:
                    self.hs[jj] = sigmoid(self.rnns[jj](x, self.hs[jj]))
            else:
                if self.hs[jj -1][0][-1] == 1 or not self.multiscale:
                    #print(self.step)
                    #print(self.freqs[jj], jj, self.step)
                    if self.binary:
                        self.hs[jj] = np.copy(binarize(sigmoid(self.rnns[jj](self.hs[jj -1], self.hs[jj]))))
                    else:
                        self.hs[jj] = sigmoid(self.rnns[jj](self.hs[jj -1], self.hs[jj]))
                    if self.multiscale:
                        self.hs[jj - 1] = self.init_h
                    

        x = np.concatenate([x] + self.hs, axis=1)


        hidden = np.tanh(np.matmul(x, self.weight[-2]) + self.bias[-2])
        action = sigmoid(np.matmul(hidden, self.weight[-1]) + self.bias[-1])
        #import ipdb; ipdb.set_trace()
        self.step += 1

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
        #self.init_h = model_params[pointer:pointer+s].reshape((1, self.hidden_size))
        #self.h = self.init_h
        for kk in range(self.n_layers):
            self.rnns[kk] = RNNCell(self.input_size, self.weight[kk], self.bias[kk])

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




def count_task(seq_len, seed):
    
    labels = np.zeros([1,seq_len])
    X =  np.random.randint(2, size=[1,seq_len])
    #X[0,:] = 1.0
    len_streak = 0
    max_streak = 0
    for ii in range(seq_len):
        if X[0,ii] == 1:
            len_streak += 1
            max_streak = max(max_streak, len_streak)
            
        else:    
            len_streak = 0
            
    labels[0, :max_streak] = 1
    return np.concatenate([X, np.zeros([1, seq_len])], axis=1), np.concatenate([np.zeros([1, seq_len]), labels], axis=1)




# USED for the GA optimization
def evluate_func(data):
    model, params, seed, n_batches =  data
    model.set_model_params(params)
    
    loss_cum = 0
    task_len = 10
    
    np.random.seed(seed)
    
    for batch in range(n_batches):
        
        model.reset()
        Xs, labels  = count_task(task_len, seed)
        for ij, (x, label) in enumerate(zip(Xs[0], labels[0])):

            x = np.array([x])
            
            #import ipdb; ipdb.set_trace()
            pred = model.get_action(x)

            if ij >= task_len:
                loss = np.abs(BCE_loss(label, pred))
                loss_cum += loss
            
    return -(loss_cum/n_batches)


# TESTS the best result
def evluate_func_test(model, params):
    
    model.set_model_params(params)
    np.random.seed(23)
    seed = 23
    
    loss_cum = 0
    task_len = 10
    
    n_batches = 20
    
    for batch in range(n_batches):
        print('-----NEW TASK------')
        model.reset()
        Xs, labels  = count_task(task_len, seed)
        for ij, (x, label) in enumerate(zip(Xs[0], labels[0])):

            x = np.array([x])
            
            #import ipdb; ipdb.set_trace()
            pred = model.get_action(x)

            if ij >= task_len:
                loss = np.abs(BCE_loss(label, pred))
                print(ij, label, pred, loss)
                loss_cum += loss
    
    print(-loss_cum/n_batches)
            
    return -(loss_cum/n_batches)

        
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--log-dir', default='', help='directory where you want to save the data')
    parser.add_argument(
        '--n-population', type=int, default=450, help='learning rate')
    parser.add_argument(
        '--seq-len',type=int, default=40, help='legth of the training sequence')
    parser.add_argument(
        '--restart-model', default=None, help='restart model file')
    parser.add_argument(
        '--n-layers',type=int, default=1, help='number of layers of the lstm')
    parser.add_argument(
        '--forget-best', action='store_true', default=False, help='GA forget best parameter')
    parser.add_argument(
        '--algo', default='ga', help='algorithm to use')
    parser.add_argument(
        '--binary', action='store_true', default=False, help='whether the hidden states of the RNN are binary')
    parser.add_argument(
        '--multiscale', action='store_true', default=False, help='whether the hidden states of the RNN are multiscale')
    
    
    
    args = parser.parse_args()

    return args

args = get_args()
args_dict = vars(args)

model = RNNModel(args.binary, args.multiscale)

if args.restart_model:
    model.load_model(args.restart_model)
    
os.system("mkdir "+ args.log_dir)
json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)
os.system("cp "+__file__+ " " + args.log_dir + "/"+__file__ )
os.system("mkdir "+ args.log_dir + "/models")


NPARAMS = model.param_count   # make this a 100-dimensinal problem.
NPOPULATION = args.n_population   # use population size of 101.
MAX_ITERATION = 4010 # run each solver for 5000 generations.

seed_width = 10
# defines genetic algorithm solver
ga = SimpleGA(NPARAMS,                # number of model parameters
               sigma_init=0.5,        # initial standard deviation
               popsize=NPOPULATION,   # population size
               elite_ratio=0.1,       # percentage of the elites
               forget_best=args.forget_best,     # forget the historical best elites
               weight_decay=0.00,     # weight decay coefficient
              )



oes = OpenES(NPARAMS,                  # number of model parameters
            sigma_init=0.5,            # initial standard deviation
            sigma_decay=0.999,         # don't anneal standard deviation
            learning_rate=0.1,         # learning rate for standard deviation
            learning_rate_decay = 1.0, # annealing the learning rate
            popsize=NPOPULATION,       # population size
            antithetic=False,          # whether to use antithetic sampling
            weight_decay=0.00,         # weight decay coefficient
            rank_fitness=False,        # use rank rather than fitness numbers
            forget_best=False)

# defines CMA-ES algorithm solver
cmaes = CMAES(NPARAMS,
              popsize=NPOPULATION,
              weight_decay=0.0,
              sigma_init = 0.5
          )

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
fit_func = evluate_func

# defines a function to use solver to solve fit_func
def test_solver(solver):
    history = []
    j = 0
    seed_width =20
    while True:
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        #print(solutions)
#         for i in range(solver.popsize):
#             fit_func((model,solutions[i]))
        seed = np.random.randint(seed_width)
        fitness_list = pool.map(fit_func, [(model,solutions[i], seed, 5) for i in range(solver.popsize)])
        
        # For the elites testing one more time on 30 seed to be sure they are really good
        elite_idxs = np.argsort(fitness_list)[::-1][0:solver.elite_popsize]
        fitness_list_2 = pool.map(fit_func, [(model,solutions[elite_idx], seed, 50) for elite_idx in elite_idxs])
        for kk,elite_idx_ in enumerate(elite_idxs):
            fitness_list[elite_idx_] = fitness_list_2[kk] 

        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j+1) % 10 == 0:
            print("fitness at iteration", (j+1), result[1])
            if args.algo == 'ga':
                print("Best:",solver.elite_rewards[0])
            print('Seed width', seed_width)
            

        if (j+1) % 100 == 0:   
            evluate_func_test(model, result[0])
            plt.plot(history)
            plt.savefig(args.log_dir + '/loss_plot.png')
            plt.close()
            
        if (j+1) % 1000 == 0:
            # save the best result
            filename = args.log_dir + '/models/model_parameters_' + str(j+1)
            with open(filename, 'wt') as out:
                res = json.dump([np.array(result[0]).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))
        
        

            
#         if  (j+1) % 40== 0 and args.algo == 'ga':
#             #print('----------------------RESET ELITES')
#             #new_elite_rewards = []
#             new_elite_params = solver.elite_params
#             new_elite_rewards = pool.map(fit_func, [(model,solver.elite_params[kk], seed, 5) for kk in range(len(solver.elite_rewards))])


#             solver = SimpleGA(NPARAMS,                # number of model parameters
#                    sigma_init=0.5,        # initial standard deviation
#                    popsize=NPOPULATION,   # population size
#                    elite_ratio=0.05,       # percentage of the elites
#                    forget_best=False,     # forget the historical best elites
#                    weight_decay=0.00,     # weight decay coefficient
#                   )
            
#             solver.elite_params = new_elite_params
#             solver.elite_rewards = new_elite_rewards
#             #print('----------------------RESET ELITES')
            
        if -result[1] <= 0.001:
            print("local optimum discovered by solver:\n", result[0])
            print("fitness score at this local optimum:", result[1])
            # save the best result
            filename = args.log_dir + '/models/model_parameters_' + str(j+1)
            with open(filename, 'wt') as out:
                res = json.dump([np.array(result[0]).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))
                seed_width += 5
                new_elite_params = solver.elite_params
            new_elite_rewards = pool.map(fit_func, [(model,solver.elite_params[kk], seed, 5) for kk in range(len(solver.elite_rewards))])


            solver = SimpleGA(NPARAMS,                # number of model parameters
                   sigma_init=0.5,        # initial standard deviation
                   popsize=NPOPULATION,   # population size
                   elite_ratio=0.05,       # percentage of the elites
                   forget_best=False,     # forget the historical best elites
                   weight_decay=0.00,     # weight decay coefficient
                  )
            
            solver.elite_params = new_elite_params
            solver.elite_rewards = new_elite_rewards
            #return history, result

        j += 1
        
             
    
if args.algo == 'ga':
    ga_history, result = test_solver(ga)
if args.algo == 'oes':
    ga_history, result = test_solver(oes)
if args.algo == 'cmaes':
    ga_history, result = test_solver(cmaes)
    
#evluate_func_test(model, result[0])
