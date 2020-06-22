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

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

#num_threads = 1 85.56 0.0021 
#num_threads = 2 85.82 0.0020
#num_threads = 4
#num_threads = 8
num_threads = 8
#num_threads = 16
#num_threads = 24

#num_threads = 1 83.63 8.20e-05
#num_threads = 2 107.65 9.50e-05
#num_threads = 4 88.05 8.00e-05
#num_threads = 8 96.84 7.07e-05
#num_threads = 12 97.72 7.09e-05
#num_threads = 16 85.04 8.15e-05
#num_threads = 24 85.46 8.4e-05
os.environ["TVM_NUM_THREADS"] = str(num_threads)


tuning_option = {
    'tuner': 'xgb',
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=100, n_parallel=8 ),
        runner=autotvm.LocalRunner(repeat=3,number=4, timeout=100),
    ),
}

def tune_kernels(N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key,
                 measure_option,
                 tuner,
                 early_stopping,
                 ):
    data =  ('TENSOR', (N, CI, H, W), 'float32')
    kernel = ('TENSOR',(CO, CI, KH, KW), 'float32')

    origin_layout = 'NCHW'

    if len(sys.argv) > 2:
        feature_type = sys.argv[2]
    else:
        #feature_type = 'datavol'
        feature_type = 'itervar'
        #feature_type = 'datavol_itervar'
    print('Feature:',feature_type)

    if len(sys.argv) > 3:
        if 'small' == sys.argv[3]:
            func_create = 'conv2d_NCHW_small.x86'
        elif 'wide' == sys.argv[3]:
            func_create = 'conv2d_NCHW_wide.x86'
        else:
            func_create = 'conv2d_NCHWc.x86'
    else:
        func_create = 'conv2d_NCHWc.x86'

    if len(sys.argv) > 4:
        count = int(sys.argv[4])
    else:
        count = 1

    if len(sys.argv) > 5:
        likwid_event=sys.argv[5]
    else:
        likwid_event=None

    task = autotvm.task.create(func_create,
                               args=(data, kernel, strides, padding, 1, origin_layout, origin_layout, 'float32'),
                               target='llvm -mcpu=core-avx2')
    using_NCHWc = True
    print(task.config_space)
    trials = min(trials, len(task.config_space))

    ctx = tvm.cpu()
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    for i in range(count): 
        log_filename = '%s_%i_%s_%s.log' % (key, i, feature_type, sys.argv[3])
        tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type, loss_type='rank', plan_size=8)
        tuner.tune(n_trial=trials,
                   measure_option=measure_option,
                   callbacks=[
                       autotvm.callback.progress_bar(trials),
                       autotvm.callback.log_to_file(log_filename)],likwid_event=likwid_event)
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
            c_np_reshape = c_np.reshape((N, CO//best_config['tile_oc'].size[-1], best_config['tile_oc'].size[-1], H, W)).transpose((0,1,3,4,2))
        a_tvm = tvm.nd.array(a_np, ctx=ctx)
        w_tvm = tvm.nd.array(w_np, ctx=ctx)
        c_tvm = tvm.nd.array(np.zeros(c_np_reshape.shape, dtype=np.float32), ctx=ctx)
        if arg_bufs[1].shape == w_tvm.shape:
            func(c_tvm, w_tvm, a_tvm)
        else:
            func(c_tvm, a_tvm, w_tvm)

        try:
            tvm.testing.assert_allclose(c_np_reshape, c_tvm.asnumpy(), rtol=1e-2)
        except:
            print('WARNING: Not equal!')
        for i in range(5):
            evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3,number=4)
            if arg_bufs[1].shape == w_tvm.shape:
                print(evaluator(c_tvm, w_tvm, a_tvm))
            else:
                print(evaluator(c_tvm, a_tvm, w_tvm))
        os.remove(log_filename)

        print(tvm.lower(s, arg_bufs, simple_mode=True))
    if likwid_event != None:
        with open('data/likwid_%s_%s_features_%icore_%i_%s.pkl' % (key, feature_type, num_threads, trials, sys.argv[3]) , 'wb') as output:
            pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/%s_%s_features_%icore_%i_%s.pkl' % (key, feature_type, num_threads, trials, sys.argv[3]) , 'wb') as output:
            pickle.dump([best_config, task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)



def tune_and_evaluate(tuning_opt):
    print("strat tuning...")

    """
    assert (len(sys.argv) == 12)
    N = int(sys.argv[1])
    H = int(sys.argv[2])
    W = int(sys.argv[3])
    CO = int(sys.argv[4])
    CI = int(sys.argv[5])
    KH = int(sys.argv[6])
    KW = int(sys.argv[7])
    strides = int(sys.argv[8])
    padding = int(sys.argv[9])
    trials = int(sys.argv[10])
    log_file = sys.argv[11]
    """


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

    key = list(benchmarks.keys())[int(sys.argv[1])]

    N, H, W, CO, CI, KH, KW = benchmarks[key]
    strides, padding, dilation =  1, 1, 1
    if KH == 1:
        padding = 0
    trials = 1024

    print("N, H, W, CO, CI, KH, KW, strides, padding \n" , N, H, W, CO, CI, KH, KW, strides, padding)
    tune_kernels(N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, key, **tuning_option)


tune_and_evaluate(tuning_option)
