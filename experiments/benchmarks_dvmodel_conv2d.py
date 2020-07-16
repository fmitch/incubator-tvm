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
from tvm.autotvm.tuner import DataVolumeTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
#from tvm.autotvm.task.topi_integration import deserialize_args
from collections import namedtuple

import argparse

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def tune_kernels(args, N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key,
                 measure_option,
                 tuner,
                 early_stopping,
                 ):
    data =  ('TENSOR', (N, CI, H, W), 'float32')
    kernel = ('TENSOR',(CO, CI, KH, KW), 'float32')

    origin_layout = 'NCHW'

    prediction_type = args.prediction
    print('Feature:', prediction_type)

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

    ctx = tvm.cpu()
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding).astype(np.float32)

    for i in range(count): 
        if random:
            log_filename = '%s_%i_%s_%s_%icore_rand.log' % (key, i, prediction_type, args.search_size,num_threads)
        else:
            log_filename = '%s_%i_%s_%s_%icore.log' % (key, i, prediction_type, args.search_size ,num_threads)


        if args.key_id != None and count == 1:
            save_ind = int(args.key_id)
        else:
            save_ind = i
        if likwid_event != None:
            if random:
                pickle_file = '/media/frost/DATA/tvm_data/fix_likwid_rand_%s_%s_features_%icore_%i_%s_%i.pkl' % (key, prediction_type, num_threads, trials, args.search_size, save_ind)
            else:
                pickle_file = '/media/frost/DATA/tvm_data/fix_likwid_%s_%s_features_%icore_%i_%s_%i.pkl' % (key, prediction_type, num_threads, trials, args.search_size, save_ind)
        else:
            if random:
                pickle_file = '/media/frost/DATA/tvm_data/fix_rand_%s_new_%s_features_%icore_%i_%s_%i.pkl' % (key, prediction_type, num_threads, trials, args.search_size, save_ind)
            else:
                pickle_file = '/media/frost/DATA/tvm_data/fix2_sa_%i_%s_new_%s_features_%icore_%i_%s_%i.pkl' % (sa_n_iter, key, prediction_type, num_threads, trials, args.search_size, save_ind)

        if os.path.exists(pickle_file):
            print('File exists', pickle_file)
            continue



        tuner = autotvm.tuner.DataVolumeTuner(task, prediction_type=prediction_type, plan_size=32, sa_n_iter=sa_n_iter)
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

        # apply history best from log file
        with autotvm.apply_history_best(log_filename):
            with tvm.target.create("llvm -mcpu=core-avx2"):
                s, arg_bufs = task.func(*task.args)
                func = tvm.build(s, arg_bufs)
        
        if using_NCHWc:
            a_np_reshape = a_np.reshape((N, CI//best_config['tile_ic'].size[-1], best_config['tile_ic'].size[-1], H, W)).transpose((0,1,3,4,2))
            w_np_reshape = w_np.reshape((CO//best_config['tile_oc'].size[-1], best_config['tile_oc'].size[-1], CI//best_config['tile_ic'].size[-1], best_config['tile_ic'].size[-1], KH, KW)).transpose((0,2,4,5,3,1))
            c_np_reshape = c_np.reshape((N, CO//best_config['tile_oc'].size[-1], best_config['tile_oc'].size[-1], H, W)).transpose((0,1,3,4,2))
        a_tvm = tvm.nd.array(a_np_reshape, ctx=ctx)
        w_tvm = tvm.nd.array(w_np_reshape, ctx=ctx)
        c_tvm = tvm.nd.array(c_np_reshape, ctx=ctx)
        if tuple(arg_bufs[1].shape) == w_tvm.shape:
            func(c_tvm, w_tvm, a_tvm)
        else:
            func(c_tvm, a_tvm, w_tvm)

        try:
            tvm.testing.assert_allclose(c_np_reshape, c_tvm.asnumpy(), rtol=1e-2)
        except:
            print('WARNING: Not equal!')
        evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3,number=4)
        if tuple(arg_bufs[1].shape) == w_tvm.shape:
            print(evaluator(c_tvm, w_tvm, a_tvm))
        else:
            print(evaluator(c_tvm, a_tvm, w_tvm))
        os.remove(log_filename)

    #print(tvm.lower(s, arg_bufs, simple_mode=True))
        if save_features:
            with open(pickle_file , 'wb') as output:
                pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)



def tune_and_evaluate():
    print("strat tuning...")


    dilation = 1;
    benchmarks = {
            'vision0':[1, 224, 224, 64, 3, 3, 3],
            'vision1':[1, 112, 112, 128, 64, 3, 3],
            'vision2':[1,  56,  56, 256, 128, 3, 3],
            'vision3':[1,  28,  28, 512, 256, 3, 3],
            'vision4':[1,  14,  14, 512, 512, 3, 3],
            'OCR1':[1,  480,  48, 16, 1, 3, 3],
            'OCR2':[1,  240,  24, 32, 16, 3, 3],
            'OCR3':[1,  120,  12, 64, 32, 3, 3],
            'OCR4':[1,   60,   6, 128, 64, 3, 3],
            'yolo0':[1, 544, 544, 32, 3, 3, 3],
            'yolo2':[1, 272, 272, 64, 32, 3, 3],
            'yolo4':[1, 136, 136, 128, 64, 3, 3],
            'yolo5':[1, 136, 136, 64, 128, 1, 1],
            'yolo7':[1,  68,  68, 256, 128, 3, 3],
            'yolo9':[1,  68,  68, 128, 256, 1, 1],
            'yolo12':[1, 34, 34, 512, 256, 3, 3],
            'yolo13':[1, 34, 34, 256, 512, 1, 1],
            'yolo17':[1, 17, 17, 1024, 512, 3, 3],
            'yolo19':[1, 17, 17, 512, 1024, 3, 3],
            'yolo23':[1, 17, 17, 28269, 1024, 1, 1]}

    parser = argparse.ArgumentParser(description='Run conv2d benchmarks in TVM')
    parser.add_argument( '-b','--benchmark', help="Which benchmark to use, int from 0-19", default=0, type=int)
    parser.add_argument( '-p','--prediction', help="Type of prediction to use, one of 'sum', 'sumL2L3', 'time'", default='sumL2L3', type=str)
    parser.add_argument( '-s','--search_size', help="Type of search space to use, one of 'small', 'mid', 'wide'", default='small', type=str)
    parser.add_argument( '-n','--num_iters', help="Int. number of times to run training", default=1, type=int)
    parser.add_argument( '-t','--trials', help="Int. Number of trials to sample", default=600, type=int)
    parser.add_argument( '-l','--likwid_event', help='Likwid event to capture during training', default=None)
    parser.add_argument( '-r','--random', help="Use Model+SA to select samples, or randomly select", default=False, action='store_true')
    parser.add_argument( '-k','--key_id', help="Key ID for RPC runner", default=None, type=str)
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
            builder=autotvm.LocalBuilder(timeout=10, n_parallel=16 ),
            runner=autotvm.LocalRunner(repeat=3,number=4, timeout=10),
            #runner=autotvm.RPCRunner('abc', '0.0.0.0', 9000, 0, 100, repeat=3,number=4, n_parallel=10),
        ),
    }


    print("N, H, W, CO, CI, KH, KW, strides, padding \n" , N, H, W, CO, CI, KH, KW, strides, padding)
    tune_kernels(args, N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key, **tuning_option)


tune_and_evaluate()
