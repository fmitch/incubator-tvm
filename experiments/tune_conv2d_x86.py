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

num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)


tuning_option = {
    'tuner': 'xgb',
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=100, n_parallel=8),
        runner=autotvm.LocalRunner(repeat=1,number=5, timeout=100),
    ),
}

def tune_kernels(N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, log_filename,
                 measure_option,
                 tuner,
                 early_stopping,
                 ):
    data =  ('TENSOR', (N, CI, H, W), 'float32')
    kernel = ('TENSOR',(CO, CI, KH, KW), 'float32')

    origin_layout = 'NCHW'

    func_create = 'conv2d_nchw_spatial_pack.dv.x86'
    task = autotvm.task.create(func_create,
                               args=(data, kernel, strides, padding, 1, 'float32'),
                               target='llvm -mcpu=core-avx2')
    using_NCHWc = False

    # Uncomment to run x86 script.
    #func_create = 'conv2d_NCHWc.x86'
    #task = autotvm.task.create(func_create,
    #                           args=(data, kernel, strides, padding, 1, origin_layout, origin_layout, 'float32'),
    #                           target='llvm -mcpu=core-avx2')
    #using_NCHWc = True

    #task.workload = ['float32', 'float32', H, W, CI, 1, CO, KH, KW, 1, 1, 1, 1]
    print(task.config_space)
    #print(len(task.config_space))
    trials = min(trials, len(task.config_space))

    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    if len(sys.argv) > 1:
        feature_type = sys.argv[1]
    else:
        feature_type = 'datavol'
        #feature_type = 'itervar'
        #feature_type = 'datavol_itervar'
    print('Feature:',feature_type)
    for i in range(1):
        tuner = autotvm.tuner.XGBTuner(task, feature_type=feature_type, loss_type='rank', plan_size=16)
        tuner.tune(n_trial=trials,
                   measure_option=measure_option,
                   callbacks=[
                       autotvm.callback.progress_bar(trials),
                       autotvm.callback.log_to_file(log_filename)])
    #with open('data/%s_features_1core_%i_n%i_%i.pkl' % (feature_type, H, N, trials) , 'wb') as output:
    #    pickle.dump([task, tuner.cost_model.saved_features], output, pickle.HIGHEST_PROTOCOL)

    dispatch_context = autotvm.apply_history_best(log_filename)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(log_filename):
        with tvm.target.create("llvm -mcpu=core-avx2"):
            s, arg_bufs = task.func(*task.args)
            func = tvm.build(s, arg_bufs)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
    
    ctx = tvm.cpu()
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
    
    if using_NCHWc:
        a_np = a_np.reshape((N, 8, H, W, CI//8))
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.array(np.zeros(c_np.shape, dtype=np.float32), ctx=ctx)
    func(c_tvm, w_tvm, a_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

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
    N, H, W, CO, CI, KH, KW, strides, padding, dilation = 1, 56, 56, 128, 64, 3, 3, 1, 1, 1
    trials = 64 
    log_file = 'conv.log'

    print("N, H, W, CO, CI, KH, KW, strides, padding \n" , N, H, W, CO, CI, KH, KW, strides, padding)
    tune_kernels(N, H, W, CO, CI, KH, KW, strides, padding, dilation, trials, log_file, **tuning_option)


tune_and_evaluate(tuning_option)
