import logging
import sys
import os
import numpy as np
from tensors import *

import pickle

import tvm
import topi
from topi.testing import conv2d_nchw_python
from tvm import te
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DataVolumeTuner
import tvm.contrib.graph_runtime as runtime
#from tvm.autotvm.task.topi_integration import deserialize_args
from collections import namedtuple

import argparse

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

num_threads = 12
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def tune_kernels(args, M, N, P, K, trials,
                 measure_option, tuner, early_stopping,):

    feature_type = args.feature
    print('Feature:', feature_type)

    count = args.num_iters
    likwid_event = args.likwid_event
    random = args.random
    sa_n_iter = args.sa_num_iters
    save_features = not (args.no_save_features)

    task = autotvm.task.create("template/tc", args=(M,N,P,K,tc_index,'float32'), target='llvm -mcpu=core-avx2')
    print(task.config_space)

    trials = min(trials, len(task.config_space))

    for i in range(count): 
        if args.key_id != None and count == 1:
            save_ind = int(args.key_id)
        else:
            save_ind = i
        if random:
            log_filename = 'tc%i_%i_%i_%s_%icore_rand.log' % (tc_index, N, save_ind, feature_type, num_threads)
        else:
            log_filename = 'tc%i_%i_%i_%s_%icore.log' % (tc_index, N, save_ind, feature_type, num_threads)

        if likwid_event != None:
            if random:
                pickle_file = '/mnt/data/tvm_data/tc/likwid_rand_tc%i_%i_%s_features_%icore_%i_%i.pkl' % (tc_index, N, feature_type, num_threads, trials,  save_ind)
            else:
                pickle_file = '/mnt/data/tvm_data/tc/likwid_tc%i_%i_%s_features_%icore_%i_%i.pkl' % (tc_index, N, feature_type, num_threads, trials,  save_ind)
        else:
            if random:
                pickle_file = '/mnt/data/tvm_data/tc/rand_tc%i_%i_%s_features_%icore_%i_%i.pkl' % (tc_index, N, feature_type, num_threads, trials,  save_ind)
            else:
                pickle_file = '/mnt/data/tvm_data/tc/tc%i_%i_new_%s_features_%icore_%i_%i.pkl' % (tc_index, N, feature_type, num_threads, trials,  save_ind)
        if os.path.exists(pickle_file):
            print('File exists', pickle_file)
            continue

        tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type, loss_type='rank',
                plan_size=32, sa_n_iter=sa_n_iter, num_threads=24)
        tuner.tune(n_trial=trials,
                   measure_option=measure_option,
                   callbacks=[
                       autotvm.callback.progress_bar(trials),
                       autotvm.callback.log_to_file(log_filename)],
                   likwid_event=likwid_event, save_features=save_features, random=random)

        dispatch_context = autotvm.apply_history_best(log_filename)
        best_config = dispatch_context.query(task.target, task.workload)
        print("\nBest config:")
        print(best_config)

    #print(tvm.lower(s, arg_bufs, simple_mode=True))
        if save_features:
            with open(pickle_file , 'wb') as output:
                pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)
        try:
            os.remove(log_filename)
        except:
            pass



def tune_and_evaluate():
    parser = argparse.ArgumentParser(description='Run conv2d benchmarks in TVM')
    parser.add_argument( '-f','--feature', help="Type of feature to use, one of 'datavol', 'itervar', 'datavol_itervar', 'itervar_silent_dv'", default='itervar', type=str)
    parser.add_argument( '-n','--num_iters', help="Int. number of times to run training", default=1, type=int)
    parser.add_argument( '-t','--trials', help="Int. Number of trials to sample", default=1000, type=int)
    parser.add_argument( '-l','--likwid_event', help='Likwid event to capture during training', default=None)
    parser.add_argument( '-r','--random', help="Use XGB+SA to select samples, or randomly select", default=False, action='store_true')
    parser.add_argument( '-k','--key_id', help="Key ID for RPC server.", default=None, type=str)
    parser.add_argument('--sa_num_iters', help="Number of iterations of simulated annealing", default=500, type=int)
    parser.add_argument('--no_save_features', help="Should save features", default=False, action='store_true')

    args = parser.parse_args()
    trials = args.trials
    global tc_index
    for size in [280,80]:
        #for ind in range(1,37):
        #for ind in [1,16,18,22,24,28,30,33,34,35,36,2]:
        for ind in [1,16,18]:
            tc_index = ind
            print("Tuning TC %i..." % tc_index)

            M,N,P,K = [size,size,size,size]

            tuning_option = {
                'tuner': 'xgb',
                'early_stopping': None,

                'measure_option': autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10, n_parallel=24 ),
                    runner=autotvm.LocalRunner(repeat=3,number=4,),
                ),
            }


            print("M,N,P,K" , M,N,P,K)
            tune_kernels(args, M,N,P,K, trials, **tuning_option)


if __name__ == "__main__":
    tune_and_evaluate()
