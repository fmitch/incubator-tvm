import logging
import time
import sys
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import random
import string

import pickle

import tvm
import topi
from topi.testing import conv2d_nchw_python
from tvm import te
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, DataVolumeTuner
import tvm.contrib.graph_runtime as runtime
#from tvm.autotvm.task.topi_integration import deserialize_args
from itertools import permutations

import argparse

import subprocess
cache_sizes = [
        int(subprocess.run(['getconf', 'LEVEL1_DCACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL2_CACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL3_CACHE_SIZE'], stdout=subprocess.PIPE).stdout) ]

#import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

global num_threads
num_threads = 32
os.environ["TVM_NUM_THREADS"] = str(num_threads)

letters = string.digits + string.ascii_letters


def get_extents_info(config, repeat=True):
    n_n = N
    ci_n = CI // config['tile_ic'].size[-1]
    co_n = CO // config['tile_oc'].size[-1]
    oh_n = H // config['tile_oh'].size[-1]
    ow_n = W // config['tile_ow'].size[-1]
    kh_n = KH 
    kw_n = KW 
    vci_n = config['tile_ic'].size[-1]
    vco_n = config['tile_oc'].size[-1]
    vh_n = config['tile_oh'].size[-1]
    vw_n = config['tile_ow'].size[-1]

    order = config['reorder_0'].perm
    extents = [n_n, co_n, oh_n, ow_n, ci_n, vci_n, kh_n, kw_n, vh_n, vw_n, vco_n]
    array_dims = [ [0,2,3,4,6,7,8,9,5], [1,4,5,6,7,10], [0,1,2,3,8,9,10] ]
    conv_dims = [ [(2,8), (6,)], [(3,9), (7,)] ]
    fastest_varying = [ [5], [10], [10]]

    if repeat:
        for arr in array_dims, fastest_varying:
            for i in range(len(arr)):
                arr[i] = (np.array(arr[i])+1).tolist()
        extents = [50] + extents
        order = [0] + (np.array(order)+1).tolist()

    return order, extents, array_dims, cache_sizes, conv_dims, fastest_varying


def get_tc_dv(ind):
    config = task.config_space.get(ind)
    d_foot, d_vol = autotvm.tuner.data_volume_estimator.estimate_dv(*get_tc_extents_info(M,N,P,K,config,tc_index))
    return -1*(d_vol[2][:,:,-1].sum(axis=0) * np.array([64/100e9, 64/44e9, 64/25e9])).sum()

def concurrency_ratio(ind):
    config = task.config_space.get(ind)

    concurrency = N*CO*H/config['tile_oc'].size[-1]/config['tile_oh'].size[-1]
    return np.floor(concurrency/num_threads) / np.ceil(concurrency/num_threads)

def get_dv(ind):
    config = task.config_space.get(ind)
    d_foot, d_vol = autotvm.tuner.data_volume_estimator.estimate_dv(*get_extents_info(config))
    return -1*(d_vol[2][:,:,-1].sum(axis=0) * np.array([64/100e9, 64/44e9, 64/25e9])).sum()

def limited_test(ind):
    tic = time.time()
    lower_llvm_limit = 1
    upper_llvm_limit = 2
    lower_asm_limit = 0.5
    upper_asm_limit = 2
    results = []
    config = task.config_space.get(ind)
    with autotvm.ApplyConfig(config):
        with tvm.target.create("llvm -mcpu=core-avx2"):
            s, arg_bufs = task.func(*task.args)
            op_func = tvm.build(s, arg_bufs)
            build_time = time.time() - tic

    ll_source = op_func.get_source()

    funcs = ll_source.split('\n\n')
    llvm_opint = 0
    asm_opint = 0
    length = 0
    for func in funcs:
        if 'fmuladd.v' in func and len(func) > length:
            length = len(func)
            longest = func

    loads = 0
    stores = 0
    fmas = 0
    if length > 0:
        lines = longest.split('\n')
        for line in lines:
            if 'load <' in line:
                loads += 1
            elif 'store <' in line:
                stores += 1
            elif 'fmuladd.v8' in line:
                fmas += 1
        if loads+stores > 0:
            llvm_opint = fmas / (loads+stores)

        if llvm_opint >= lower_llvm_limit and llvm_opint <= upper_llvm_limit:
            tic = time.time()
            asm_source = op_func.get_source('asm')
            asm_time = time.time() - tic


            funcs = asm_source.split(':\n')
            length = 0
            for func in funcs:
                if 'vfmadd' in func and len(func) > length:
                    length = len(func)
                    longest = func
            moves = 0
            fmas = 0
            if length > 0:
                lines = longest.split('\n')
                for line in lines:
                    if 'vmov' in line and 'ymm' in line:
                        moves += 1
                    elif 'vfmadd' in line and 'ymm' in line:
                        fmas += 1
                        if '(%r' in line:
                            moves += 1
            if moves > 0:
                asm_opint = fmas / moves

            if asm_opint >= lower_asm_limit and asm_opint <= upper_asm_limit:
                module_file = os.path.join('/tmp/', ''.join(random.choice(letters) for i in range(10)) + '.o')
                op_func.save(module_file)
                return module_file, llvm_opint, asm_opint, ind,build_time, asm_time

    return '', llvm_opint, asm_opint, ind, build_time, 0


def eval_time(ind, module_file):
    config = task.config_space.get(ind)
    with autotvm.ApplyConfig(config):
        with tvm.target.create("llvm -mcpu=core-avx2"):
            s, arg_bufs = task.func(*task.args)
            func = tvm.runtime.load_module(module_file)

        d_shape = (N, CI//config['tile_ic'].size[-1], H,W, config['tile_ic'].size[-1])
        k_shape = (CO//config['tile_oc'].size[-1], CI//config['tile_ic'].size[-1],
                KH,KW, config['tile_ic'].size[-1], config['tile_oc'].size[-1]) 
        o_shape = (N, CO//config['tile_oc'].size[-1], H,W, config['tile_oc'].size[-1])
        a_np = np.random.uniform(size=d_shape)
        b_np = np.random.uniform(size=k_shape)
        c_np = np.zeros(o_shape)
        ctx = tvm.cpu()
        a_tvm = tvm.nd.array(a_np.astype(np.float32), ctx=ctx)
        b_tvm = tvm.nd.array(b_np.astype(np.float32), ctx=ctx)
        c_tvm = tvm.nd.array(c_np.astype(np.float32), ctx=ctx)

        evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10,number=4,)
        variation = 1
        while variation > 0.05:
            if tuple(arg_bufs[1].shape) == b_tvm.shape:
                res = np.array(sorted(evaluator(c_tvm, b_tvm, a_tvm).results)[:-5])
            else:
                res = np.array(sorted(evaluator(c_tvm, a_tvm, b_tvm).results)[:-5])
            variation = res.std() / res.mean()

        #if tuple(arg_bufs[1].shape) == b_tvm.shape:
        #    res = evaluator(c_tvm, b_tvm, a_tvm)
        #else:
        #    res = evaluator(c_tvm, a_tvm, b_tvm)

        return res.mean(), ind

def tune_kernels(args, trials, key, cr_limit):
    data =  ('TENSOR', (N, CI, H, W), 'float32')
    kernel = ('TENSOR',(CO, CI, KH, KW), 'float32')

    origin_layout = 'NCHW'

    feature_type = 'itervar'
    print('Feature:',feature_type)

    func_create = 'conv2d_NCHWc_huge.x86'

    global task
    task = autotvm.task.create(func_create, 
            args=(data, kernel, strides, padding, 1, origin_layout, origin_layout, 'float32'),
            target='llvm -mcpu=core-avx2')

    if 'NCHWc' in func_create:
        using_NCHWc = True
    else:
        using_NCHWc = False 
    
    print(task.config_space)
    outer_trials = min(int(1e10), len(task.config_space))
    trials = min(trials, len(task.config_space))


    pickle_file = 'data/conv/perm%.2f_timed_asm_%s_%icore_%i.pkl' % (cr_limit, key, num_threads, trials)
    if os.path.exists(pickle_file):
        print('File exists', pickle_file)
        return
        with open(pickle_file, 'rb') as fi:
            inds, res, dv, res_times, asm, llvm = pickle.load(fi)
        best = np.array(res).mean(axis=1).argsort()
        inds = np.array(inds)
        cr = []
        for ind in inds:
            cr.append(concurrency_ratio(ind))
        cr = np.array(cr)
        res = np.array(res).mean(axis=1)
        print(res[best[:10]])
        print(np.array(asm)[best[:10]])
        print(np.array(llvm)[best[:10]])
        print(cr[best[:10]])
        #for ind in inds[best[:10]]:
        #    print(task.config_space.get(ind))
        return

    pool_threads = 80#cpu_count()

    #configs = np.random.choice(len(task.config_space), size=outer_trials, replace=False)
    configs = range(outer_trials)

    print('Running Data Volume model...')
    tic = time.time()
    with Pool(pool_threads) as p:
        cr = p.map(concurrency_ratio, configs)
    print('CR for %i configs: %f' % (len(configs), time.time() - tic))
    cr = np.array(cr)
    configs = np.array(configs)[(cr > cr_limit)]
    cr = np.array(cr)[(cr > cr_limit)]

    with Pool(pool_threads) as p:
        dv = p.map(get_dv, configs)
    print('DV for %i configs: %f' % (len(configs), time.time() - tic))

    dv = -1*np.array(dv)
    dv_order = dv.argsort()
    configs = configs[dv_order]
    dv = dv[dv_order]
    num_configs = len(configs)
    dv_dict = dict(zip(configs,dv))

    best_flops = 0.0
    flops = 0.0
    counter = 0
    print('Running on hardware...')
    sorted_order = np.array(dv).argsort()
    vec_counter = 0
    to_try = np.array(configs)[sorted_order]
    build_counter = 0

    inds = []
    results = []
    dv = []
    asm_opints = []
    llvm_opints = []
    result_times = []

    asm_times = 0
    while len(results) < trials and build_counter < num_configs:
        inds_to_test = []
        module_files = []
        start_index = build_counter

        with Pool(pool_threads) as p:
            for module_file, llvm, asm, ind, build_time, asm_time in p.map(limited_test, to_try[start_index:start_index+100*pool_threads]):
        #for ind in to_try:
        #        should_test, ind = limited_test(ind)
                build_counter += 1
                if len(module_file) > 0:
                    llvm_opints.append(llvm)
                    asm_opints.append(asm)
                    inds_to_test.append(ind)
                    module_files.append(module_file)
                    vec_counter += 1
                #print('Prepping tests: %.2f/%.2f GFLOPS %i/%i  (%i),    %.1f s               \r' % 
                #        (flops, best_flops, counter, num_configs,
                #            build_counter, time.time()-tic), end='')

        #finished_index = np.where(to_try == inds_to_test[-1])[0][0]
        #to_try = to_try[finished_index+1:]

        #with Pool(6) as p:
        #    for x, ind in p.imap(limited_test, to_try):
        inds_to_test = np.array(inds_to_test)
        for ind, module_file in zip(inds_to_test, module_files):
                x, ind = eval_time(ind, module_file)
                result_times.append(time.time() - tic)
                counter += 1
                mean_time = np.array(x).mean()
                flops = task.flop/(mean_time*1e9)
                best_flops = max(flops, best_flops)
                if best_flops == flops:
                    best_ind = ind
                inds.append(ind)
                results.append(x)
                dv.append(dv_dict[ind])
                print('Testing: %.2f/%.2f GFLOPS %i/%i  (%i),    %.1f s               \r' % 
                        (flops, best_flops, counter, num_configs, 
                            build_counter, time.time()-tic), end='')
                os.remove(module_file)
                os.remove(module_file+'.so')

    print()
    print('Best config:', task.config_space.get(best_ind), best_flops)
    print('Saving %s' % pickle_file)
    with open(pickle_file, 'wb') as output:
        pickle.dump([inds, results, dv, result_times, asm_opints, llvm_opints],
            output, pickle.HIGHEST_PROTOCOL)
    return

def tune_and_evaluate():

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

    parser = argparse.ArgumentParser(description='Run TC benchmarks in TVM')
    parser.add_argument( '-t','--trials', help="Int. Number of trials to sample", default=2000, type=int)
    parser.add_argument( '-b','--benchmark', help="Int. Number of Tensor Contraction benchmark (0-19)", default=0, type=int)

    args = parser.parse_args()
    trials = args.trials
    key = list(benchmarks.keys())[args.benchmark]

    global N, H, W, CO, CI, KH, KW 
    global strides, padding, dilation
    N, H, W, CO, CI, KH, KW = benchmarks[key]
    strides, padding, dilation =  1, 1, 1
    if KH == 1:
        padding = 0

    cr_limit = 0.9

    print("N, H, W, CO, CI, KH, KW, strides, padding \n" , N, H, W, CO, CI, KH, KW, strides, padding)
    tune_kernels(args, trials, key, cr_limit)


if __name__ == "__main__":
    tune_and_evaluate()


