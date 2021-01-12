import numpy as np

# Generator of randomized test sequences


def count_task(seq_len):

    labels = np.zeros([1,seq_len])
    X =  np.random.randint(2, size=[1,seq_len])
    #X[0,:] = 1.0
    len_streak = 0
    max_streak = 0
    for ii in range(seq_len):
        if X[0,ii] == 1:
            len_streak += 1
            max_streak = max(max_streak, len_streak)
            
#         else:    
#             len_streak = 0
            
    labels[0, :max_streak] = 1
    return np.concatenate([X, np.zeros([1, seq_len])], axis=1), np.concatenate([np.zeros([1, seq_len]), labels], axis=1)

def dummi_task(sid):
    #np.random.seed(sid)
    return np.random.randint(10, size = 5)


if __name__ == '__main__':

    for kk in range(100):
        print(count_task(10))
#     np.random.seed(0)
#     print('------SEED:',0 )
#     for jj in range(10):
#         print(np.random.randint(10, size = 5))
        
#     np.random.seed(0)
#     print('------SEED:', 0 )
#     for jj in range(10):
#         print(np.random.randint(10, size = 5))
        
#     np.random.seed(11)
#     print('------SEED:',11 )
#     for jj in range(10):
#         print(np.random.randint(10, size = 5))
   
#     np.random.seed(11)
#     print('------SEED:',11 )
#     for jj in range(10):
#         print(np.random.randint(10, size = 5))
    
# #     import multiprocessing as mp
# #     pool = mp.Pool(mp.cpu_count())
# #     sids = [1,1,1,1,1,1,1,1,4,1]
# #     resu_list = pool.map(dummi_task, [(i) for i in sids])
# #     print(resu_list)
    