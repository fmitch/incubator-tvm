import logging
import sys
import os
import numpy as np

import pickle

import tvm
import topi
from topi.testing import conv2d_nchw_python
from tvm import te
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
#from tvm.autotvm.task.topi_integration import deserialize_args
from collections import namedtuple

import argparse

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

num_threads = 32
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def tune_kernels(args, N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key,
                 measure_option,
                 tuner,
                 early_stopping,
                 ):
    data =  ('TENSOR', (N, CI, H, W), 'float32')
    kernel = ('TENSOR',(CO, CI, KH, KW), 'float32')

    origin_layout = 'NCHW'

    feature_type = args.feature
    print('Feature:',feature_type)

    if 'small' == args.search_size:
        func_create = 'conv2d_NCHWc_small.x86'
    elif 'mid' == args.search_size:
        func_create = 'conv2d_NCHWc_mid.x86'
    elif 'wide' == args.search_size:
        func_create = 'conv2d_NCHWc_wide.x86'
    elif 'huge' == args.search_size:
        func_create = 'conv2d_NCHWc_huge.x86'
    #elif 'nchw_small' == args.search_size:
    #    func_create = 'conv2d_NCHW_small.x86'
    #elif 'nchw_mid' == args.search_size:
    #    func_create = 'conv2d_NCHW_mid.x86'
    #elif 'nchw_wide' == args.search_size:
    #    func_create = 'conv2d_NCHW_wide.x86'
    else:
        func_create = 'conv2d_NCHWc.x86'

    count = args.num_iters
    likwid_event = args.likwid_event
    random = args.random
    sa_n_iter = args.sa_num_iters
    save_features = not (args.no_save_features)

    task = autotvm.task.create(func_create,
                               args=(data, kernel, strides, padding, 1, origin_layout, origin_layout, 'float32'),
                               target='llvm -mcpu=core-avx2')
    if 'NCHWc' in func_create:
        using_NCHWc = True
    else:
        using_NCHWc = False 
    print(task.config_space)
    trials = min(trials, len(task.config_space))

    for i in range(count): 
        if random:
            log_filename = '%s_%i_%s_%s_%icore_rand.log' % (key, i, feature_type, args.search_size,num_threads)
        else:
            log_filename = '%s_%i_%s_%s_%icore.log' % (key, i, feature_type, args.search_size ,num_threads)

        if args.key_id != None and count == 1:
            save_ind = int(args.key_id)
        else:
            save_ind = i
        if likwid_event != None:
            if random:
                pickle_file = 'data/conv/likwid_rand_%s_%s_features_%icore_%i_%s_%i.pkl' % (key, feature_type, num_threads, trials, args.search_size, save_ind)
            else:
                pickle_file = 'data/conv/likwid_%s_%s_features_%icore_%i_%s_%i.pkl' % (key, feature_type, num_threads, trials, args.search_size, save_ind)
        else:
            if random:
                pickle_file = 'data/conv/rand_%s_%s_features_%icore_%i_%s_%i.pkl' % (key, feature_type, num_threads, trials, args.search_size, save_ind)
            else:
                pickle_file = 'data/conv/%s_%s_features_%icore_%i_%s_%i.pkl' % (key, feature_type, num_threads, trials, args.search_size, save_ind)
        if os.path.exists(pickle_file):
            print('File exists', pickle_file)
            continue

        tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type, loss_type='rank', plan_size=80, sa_n_iter=sa_n_iter)
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

        try:
            os.remove(log_filename)
        except:
            pass

        if save_features:
            with open(pickle_file , 'wb') as output:
                pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)



def tune_and_evaluate():
    print("Start tuning...")


    dilation = 1;
    benchmarks = {
            #'vision0':[1, 224, 224, 64, 3, 3, 3],
            #'vision1':[1, 112, 112, 128, 64, 3, 3],
            #'vision2':[1,  56,  56, 256, 128, 3, 3],
            #'vision3':[1,  28,  28, 512, 256, 3, 3],
            #'vision4':[1,  14,  14, 512, 512, 3, 3],
            #'OCR1':[1,  480,  48, 16, 1, 3, 3],
            #'OCR2':[1,  240,  24, 32, 16, 3, 3],
            #'OCR3':[1,  120,  12, 64, 32, 3, 3],
            #'OCR4':[1,   60,   6, 128, 64, 3, 3],
            'yolo0':[1, 544, 544, 32, 3, 3, 3],
            'yolo2':[1, 272, 272, 64, 32, 3, 3],
            'yolo2_L3':[2, 272, 272, 64, 32, 3, 3],
            'yolo4':[1, 136, 136, 128, 64, 3, 3],
            'yolo4_L3':[4, 136, 136, 128, 64, 3, 3],
            'yolo5':[1, 136, 136, 64, 128, 1, 1],
            'yolo5_L3':[4, 136, 136, 64, 128, 1, 1],
            'yolo7':[1,  68,  68, 256, 128, 3, 3],
            'yolo7_L3':[8,  68,  68, 256, 128, 3, 3],
            'yolo9':[1,  68,  68, 128, 256, 1, 1],
            'yolo9_L3':[8,  68,  68, 128, 256, 1, 1],
            'yolo12':[1, 34, 34, 512, 256, 3, 3],
            'yolo12_L3':[16, 34, 34, 512, 256, 3, 3],
            'yolo13':[1, 34, 34, 256, 512, 1, 1],
            'yolo13_L3':[16, 34, 34, 256, 512, 1, 1],
            'yolo17':[1, 17, 17, 1024, 512, 3, 3],
            'yolo17_L3':[32, 17, 17, 1024, 512, 3, 3],
            'yolo19':[1, 17, 17, 512, 1024, 1, 1],
            'yolo19_L3':[32, 17, 17, 512, 1024, 1, 1],
            'yolo23':[1, 17, 17, 28269, 1024, 1, 1],
            }

    parser = argparse.ArgumentParser(description='Run conv2d benchmarks in TVM')
    parser.add_argument( '-b','--benchmark', help="Which benchmark to use, int from 0-19", default=0, type=int)
    parser.add_argument( '-f','--feature', help="Type of feature to use, one of 'datavol', 'itervar', 'datavol_itervar', 'itervar_silent_dv'", default='itervar', type=str)
    parser.add_argument( '-s','--search_size', help="Type of search space to use, one of 'small', 'mid', 'wide'", default='huge', type=str)
    parser.add_argument( '-n','--num_iters', help="Int. number of times to run training", default=1, type=int)
    parser.add_argument( '-t','--trials', help="Int. Number of trials to sample", default=2000, type=int)
    parser.add_argument( '-l','--likwid_event', help='Likwid event to capture during training', default=None)
    parser.add_argument( '-r','--random', help="Use XGB+SA to select samples, or randomly select", default=False, action='store_true')
    parser.add_argument( '-k','--key_id', help="Key ID for RPC server.", default=None, type=str)
    parser.add_argument('--sa_num_iters', help="Number of iterations of simulated annealing", default=500, type=int)
    parser.add_argument('--no_save_features', help="Should save features", default=False, action='store_true')

    args = parser.parse_args()
    trials = args.trials

    key = list(benchmarks.keys())[args.benchmark]

    N, H, W, CO, CI, KH, KW = benchmarks[key]
    strides, padding, dilation =  1, 1, 1
    if KH == 1:
        padding = 0

    tuning_option = {
        'tuner': 'xgb',
        'early_stopping': None,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10, n_parallel=80 ),
            runner=autotvm.LocalRunner(repeat=10,number=4),
        ),
    }


    print("N, H, W, CO, CI, KH, KW, strides, padding \n" , N, H, W, CO, CI, KH, KW, strides, padding)
    tune_kernels(args, N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key, **tuning_option)


tune_and_evaluate()
