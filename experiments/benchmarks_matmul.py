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
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DataVolumeTuner
import tvm.contrib.graph_runtime as runtime
#from tvm.autotvm.task.topi_integration import deserialize_args
from collections import namedtuple
from itertools import permutations

import argparse

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

@autotvm.template("template/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    yo, yi = cfg.define_split("tile_y", y, num_outputs=2, filter=lambda y: y.size[-1] <= 128, policy='verbose')
    xo, xi = cfg.define_split("tile_x", x, num_outputs=2, filter=lambda y: y.size[-1] <= 128, policy='verbose')
    ko, ki = cfg.define_split("tile_k", k, num_outputs=2, filter=lambda y: y.size[-1] <= 128, policy='verbose')

    order = [yo, xo, ko, yi, xi, ki]
    perms = []
    for start in permutations([yo, xo, ko]):
        for end in permutations([yi, xi, ki]):
            perms.append(list(start)+list(end))
    cfg.define_reorder('reorder_0', order,
            policy='candidate', candidate=perms)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    ko, ki = cfg["tile_k"].apply(s, C, k)

    order = [yo, xo, ko, yi, xi, ki]
    cfg['reorder_0'].apply(s, C, order)

    yo_value = y.dom.extent.value // cfg['tile_y'].size[-1]
    xo_value = x.dom.extent.value // cfg['tile_x'].size[-1]
    ko_value = k.dom.extent.value // cfg['tile_k'].size[-1]
    yi_value = cfg['tile_y'].size[-1]
    xi_value = cfg['tile_x'].size[-1]
    ki_value = cfg['tile_k'].size[-1]
    cfg.extents = [yo_value, xo_value, ko_value, yi_value, xi_value, ki_value]
    cfg.array_dims = [ [0, 2, 3, 5], [1,2,5,4], [0,1,3,4]]
    cfg.conv_dims = []
    cfg.fastest_varying = [[5],[4],[4]]
    cfg.arrays = ['A', 'B', 'C']

    return s, [A, B, C]




def tune_kernels(args, N, L, M, trials,
                 measure_option,
                 tuner,
                 early_stopping,
                 ):

    feature_type = args.feature
    print('Feature:', feature_type)

    count = args.num_iters
    likwid_event = args.likwid_event
    random = args.random
    sa_n_iter = args.sa_num_iters
    save_features = not (args.no_save_features)

    task = autotvm.task.create("template/matmul", args=(N, L, M, 'float32'), target='llvm -mcpu=core-avx2')
    print(task.config_space)

    trials = min(trials, len(task.config_space))

    for i in range(count): 
        if args.key_id != None and count == 1:
            save_ind = int(args.key_id)
        else:
            save_ind = i
        if random:
            log_filename = 'matmul_%i_%i_%i_%s_%icore_rand.log' % (N, L, save_ind, feature_type, num_threads)
        else:
            log_filename = '%i_%i_%i_%s_%icore.log' % (N, L, save_ind, feature_type, num_threads)

        if likwid_event != None:
            if random:
                pickle_file = '/media/frost/DATA/tvm_data/matmul/likwid_matmul_rand_%i_%i_%s_features_%icore_%i_%i.pkl' % (N, L, feature_type, num_threads, trials, save_ind)
            else:
                pickle_file = '/media/frost/DATA/tvm_data/matmul/likwid_matmul_%i_%s_features_%icore_%i_%i.pkl' % (N, feature_type, num_threads, trials, save_ind)
        else:
            if random:
                pickle_file = '/media/frost/DATA/tvm_data/matmul/matmul_rand_%i_%s_features_%icore_%i_%i.pkl' % (N, feature_type, num_threads, trials,  save_ind)
            else:
                pickle_file = '/media/frost/DATA/tvm_data/matmul/matmul_%i_%i_%s_features_%icore_%i_%i.pkl' % (N, L, feature_type, num_threads, trials,  save_ind)
        if os.path.exists(pickle_file):
            print('File exists', pickle_file)
            continue

        if feature_type == 'time':
            tuner = autotvm.tuner.DataVolumeTuner(task, prediction_type=feature_type,
                    plan_size=32, sa_n_iter=sa_n_iter)
        else:
            tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type, loss_type='rank',
                    plan_size=32, sa_n_iter=sa_n_iter)
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


        with autotvm.apply_history_best(log_filename):
            with tvm.target.create("llvm -mcpu=core-avx2"):
                s, arg_bufs = matmul(N, L, M, 'float32')
                func = tvm.build(s, arg_bufs)

        a_np = np.random.uniform(size=(N, L)).astype(np.float32)
        b_np = np.random.uniform(size=(L, M)).astype(np.float32)
        c_np = a_np.dot(b_np)

        c_tvm = tvm.nd.empty(c_np.shape)
        func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

        try:
            tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
        except:
            print('WARNING: Not equal!')

        ctx = tvm.cpu()
        evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3,number=4)
        print(evaluator(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm))
        os.remove(log_filename)

    #print(tvm.lower(s, arg_bufs, simple_mode=True))
        if save_features:
            with open(pickle_file , 'wb') as output:
                pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)



def tune_and_evaluate():
    print("strat tuning...")


    dilation = 1;
    benchmarks = {
            0: [2000, 20],
            1: [2000, 200],
            2: [2000, 2000],
            3: [1000, 20],
            4: [1000, 200],
            5: [1000, 2000]
            }

    parser = argparse.ArgumentParser(description='Run conv2d benchmarks in TVM')
    parser.add_argument( '-b','--benchmark', help="Which benchmark to use, int from 0-19", default=0, type=int)
    parser.add_argument( '-f','--feature', help="Type of feature to use, one of 'datavol', 'itervar', 'datavol_itervar', 'itervar_silent_dv'", default='itervar', type=str)
    parser.add_argument( '-n','--num_iters', help="Int. number of times to run training", default=1, type=int)
    parser.add_argument( '-t','--trials', help="Int. Number of trials to sample", default=600, type=int)
    parser.add_argument( '-l','--likwid_event', help='Likwid event to capture during training', default=None)
    parser.add_argument( '-r','--random', help="Use XGB+SA to select samples, or randomly select", default=False, action='store_true')
    parser.add_argument( '-k','--key_id', help="Key ID for RPC server.", default=None, type=str)
    parser.add_argument('--sa_num_iters', help="Number of iterations of simulated annealing", default=500, type=int)
    parser.add_argument('--no_save_features', help="Should save features", default=False, action='store_true')

    args = parser.parse_args()
    trials = args.trials

    key = list(benchmarks.keys())[args.benchmark]

    N, L = benchmarks[key]
    strides, padding, dilation =  1, 1, 1

    tuning_option = {
        'tuner': 'xgb',
        'early_stopping': None,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=20, n_parallel=32 ),
            runner=autotvm.LocalRunner(repeat=3,number=4, timeout=20),
        ),
    }


    print("N, L" , N, L)
    M = N
    tune_kernels(args, N, L, M, trials, **tuning_option)


if __name__ == "__main__":
    tune_and_evaluate()
