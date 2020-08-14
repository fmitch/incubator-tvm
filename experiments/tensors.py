import numpy as np
from itertools import permutations

from tvm import te
from tvm import autotvm

import subprocess
cache_sizes = [
        int(subprocess.run(['getconf', 'LEVEL1_DCACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL2_CACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL3_CACHE_SIZE'], stdout=subprocess.PIPE).stdout)]

def get_matmul_extents_info(M,N,K,config,matmul_index,repeat=True):
    array_dict = {
            1: ( [ [0,2,3,5], [1,2,4,5], [0,1,3,4]], [[2,5],[2,5],[1,4]] ),
            2: ( [ [0,2,3,5], [1,2,5,4], [0,1,3,4]], [[2,5],[1,4],[1,4]] ),
            3: ( [ [0,2,5,3], [1,2,4,5], [0,1,3,4]], [[0,3],[2,5],[1,4]] ),
            4: ( [ [0,2,5,3], [1,2,5,4], [0,1,3,4]], [[0,3],[1,4],[1,4]] ),
            }

    mo_value = np.ceil(M / config['tile_m'].size[-1])
    no_value = np.ceil(N / config['tile_n'].size[-1])
    ko_value = np.ceil(K / config['tile_k'].size[-1])
    mi_value = config['tile_m'].size[-1]
    ni_value = config['tile_n'].size[-1]
    ki_value = config['tile_k'].size[-1]
    order = config['reorder_0'].perm
    extents = [mo_value, no_value, ko_value, mi_value, ni_value, ki_value]
    array_dims, fastest_varying = array_dict[matmul_index]
    conv_dims = []
    arrays = ['A', 'B', 'C']

    if repeat:
        for arr in array_dims, fastest_varying:
            for i in range(len(arr)):
                arr[i] = (np.array(arr[i])+1).tolist()
        extents = [50] + extents
        order = [0] + (np.array(order)+1).tolist()

    return order, extents, array_dims, cache_sizes, conv_dims, fastest_varying, arrays

@autotvm.template("template/matmul")
def matmul(M,N,K,matmul_index, dtype):
    A = te.placeholder((K,M), name='A', dtype=dtype)
    B = te.placeholder((N,K), name='B', dtype=dtype)

    k = te.reduce_axis((0, K), name='k')

    lambda_dict = { 
            1:lambda m,n: te.sum(A[m,k] * B[n,k], axis=k),
            2:lambda m,n: te.sum(A[m,k] * B[k,n], axis=k),
            3:lambda m,n: te.sum(A[k,m] * B[n,k], axis=k),
            4:lambda m,n: te.sum(A[k,m] * B[k,n], axis=k),
            }

    return matmul_template(M,N,K, A,B,k, lambda_dict[matmul_index], matmul_index, dtype)

def matmul_template(M,N,K, A,B,k,lambda_func, matmul_index, dtype):
    C = te.compute((M,N), lambda_func, name='C')
    s = te.create_schedule(C.op)

    # schedule
    m,n = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    mo, mi = cfg.define_split("tile_m", m, num_outputs=2, policy='verbose')
    no, ni = cfg.define_split("tile_n", n, num_outputs=2, policy='verbose')
    ko, ki = cfg.define_split("tile_k", k, num_outputs=2, policy='verbose')

    order = [mo, no, ko, mi, ni, ki] 
    perms = []
    #for start in permutations([mo, no, po, ko]):
    start = [mo, no, ko]
    for end in permutations([mi, ni, ki]):
        perms.append(list(start)+list(end))

    cfg.define_reorder('reorder_0', order,
            policy='candidate', candidate=perms)
    ##### define space end #####

    # schedule according to config
    mo, mi = cfg["tile_m"].apply(s, C, m)
    no, ni = cfg["tile_n"].apply(s, C, n)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    order = [mo, no, ko, mi, ni, ki] 
    cfg['reorder_0'].apply(s, C, order)

    #to_parallel = []
    #for i in cfg['reorder_0'].perm:
    #    if i == 3:
    #        break
    #    to_parallel.append(order[i])
    #parallel_axis = s[C].fuse(*to_parallel)

    parallel_axis = s[C].fuse(mo,no)
    s[C].parallel(parallel_axis)

    _, cfg.extents, cfg.array_dims, _, cfg.conv_dims, cfg.fastest_varying, cfg.arrays = get_matmul_extents_info(M,N,K,cfg, matmul_index)

    return s, [A, B, C]











def get_tc_extents_info(M,N,P,K,config,tc_index,repeat=True):
    array_dict = {
            1: ( [ [0,3,7,4], [1,2,3,5,6,7], [0,1,2,5,6,4]], [[0,4],[3,7],[0,4]] ),
            2: ( [ [0,3,7,4], [1,2,3,5,6,7], [0,1,2,5,6,4]], [[0,4],[3,7],[0,4]] ),
            3: ( [ [0,3,7,4], [1,2,3,7,6,5], [0,1,2,5,6,4]], [[0,4],[1,5],[0,4]] ),
            4: ( [ [0,3,7,4], [1,2,3,5,7,6], [0,1,2,5,6,4]], [[0,4],[2,6],[0,4]] ),
            5: ( [ [0,3,7,4], [1,2,3,7,6,5], [0,1,2,5,6,4]], [[0,4],[1,5],[0,4]] ),
            6: ( [ [0,3,7,4], [1,2,3,5,7,6], [0,1,2,5,6,4]], [[0,4],[2,6],[0,4]] ),

            7:  ( [ [0,4,3,7], [1,2,3,5,6,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            8:  ( [ [0,4,3,7], [1,2,3,5,6,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            9:  ( [ [0,4,3,7], [1,2,3,7,6,5], [0,1,2,5,6,4]], [[3,7],[1,5],[0,4]] ),
            10: ( [ [0,4,3,7], [1,2,3,5,7,6], [0,1,2,5,6,4]], [[3,7],[2,6],[0,4]] ),
            11: ( [ [0,4,3,7], [1,2,3,7,6,5], [0,1,2,5,6,4]], [[3,7],[1,5],[0,4]] ),
            12: ( [ [0,4,3,7], [1,2,3,5,7,6], [0,1,2,5,6,4]], [[3,7],[2,6],[0,4]] ),

            13: ( [ [1,3,7,5], [0,2,3,4,6,7], [0,1,2,5,6,4]], [[1,5],[3,7],[0,4]] ),
            14: ( [ [1,3,7,5], [0,2,3,4,6,7], [0,1,2,5,6,4]], [[1,5],[3,7],[0,4]] ),
            15: ( [ [1,3,7,5], [0,2,3,7,6,4], [0,1,2,5,6,4]], [[1,5],[0,4],[0,4]] ),
            16: ( [ [1,3,7,5], [0,2,3,4,7,6], [0,1,2,5,6,4]], [[1,5],[2,6],[0,4]] ),
            17: ( [ [1,3,7,5], [0,2,3,7,6,4], [0,1,2,5,6,4]], [[1,5],[0,4],[0,4]] ),
            18: ( [ [1,3,7,5], [0,2,3,4,7,6], [0,1,2,5,6,4]], [[1,5],[2,6],[0,4]] ),
                
            19: ( [ [1,5,3,7], [0,2,3,4,6,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            20: ( [ [1,5,3,7], [0,2,3,4,6,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            21: ( [ [1,5,3,7], [0,2,3,7,6,4], [0,1,2,5,6,4]], [[3,7],[0,4],[0,4]] ),
            22: ( [ [1,5,3,7], [0,2,3,4,7,6], [0,1,2,5,6,4]], [[3,7],[2,6],[0,4]] ),
            23: ( [ [1,5,3,7], [0,2,3,7,6,4], [0,1,2,5,6,4]], [[3,7],[0,4],[0,4]] ),
            24: ( [ [1,5,3,7], [0,2,3,4,7,6], [0,1,2,5,6,4]], [[3,7],[2,6],[0,4]] ),
                
            25: ( [ [2,3,7,6], [0,1,3,4,5,7], [0,1,2,5,6,4]], [[2,6],[3,7],[0,4]] ),
            26: ( [ [2,3,7,6], [0,1,3,4,5,7], [0,1,2,5,6,4]], [[2,6],[3,7],[0,4]] ),
            27: ( [ [2,3,7,6], [0,1,3,7,5,4], [0,1,2,5,6,4]], [[2,6],[0,4],[0,4]] ),
            28: ( [ [2,3,7,6], [0,1,3,4,7,5], [0,1,2,5,6,4]], [[2,6],[1,5],[0,4]] ),
            29: ( [ [2,3,7,6], [0,1,3,7,5,4], [0,1,2,5,6,4]], [[2,6],[0,4],[0,4]] ),
            30: ( [ [2,3,7,6], [0,1,3,4,7,5], [0,1,2,5,6,4]], [[2,6],[1,5],[0,4]] ),
                
            31: ( [ [2,6,3,7], [0,1,3,4,5,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            32: ( [ [2,6,3,7], [0,1,3,4,5,7], [0,1,2,5,6,4]], [[3,7],[3,7],[0,4]] ),
            33: ( [ [2,6,3,7], [0,1,3,7,5,4], [0,1,2,5,6,4]], [[3,7],[0,4],[0,4]] ),
            34: ( [ [2,6,3,7], [0,1,3,4,7,5], [0,1,2,5,6,4]], [[3,7],[1,5],[0,4]] ),
            35: ( [ [2,6,3,7], [0,1,3,7,5,4], [0,1,2,5,6,4]], [[3,7],[0,4],[0,4]] ),
            36: ( [ [2,6,3,7], [0,1,3,4,7,5], [0,1,2,5,6,4]], [[3,7],[1,5],[0,4]] ),
            }

    mo_value = np.ceil(M / config['tile_m'].size[-1])
    no_value = np.ceil(N / config['tile_n'].size[-1])
    po_value = np.ceil(P / config['tile_p'].size[-1])
    ko_value = np.ceil(K / config['tile_k'].size[-1])
    mi_value = config['tile_m'].size[-1]
    ni_value = config['tile_n'].size[-1]
    pi_value = config['tile_p'].size[-1]
    ki_value = config['tile_k'].size[-1]
    order = config['reorder_0'].perm
    extents = [mo_value, no_value, po_value, ko_value, mi_value, ni_value, pi_value, ki_value]
    array_dims, fastest_varying = array_dict[tc_index]
    conv_dims = []
    arrays = ['A', 'B', 'C']

    if repeat:
        for arr in array_dims, fastest_varying:
            for i in range(len(arr)):
                arr[i] = (np.array(arr[i])+1).tolist()
        extents = [50] + extents
        order = [0] + (np.array(order)+1).tolist()

    return order, extents, array_dims, cache_sizes, conv_dims, fastest_varying, arrays

@autotvm.template("template/tc")
def tc(M,N,P,K,tc_index, dtype):
    A = te.placeholder((K,M), name='A', dtype=dtype)
    B = te.placeholder((P,N,K), name='B', dtype=dtype)

    k = te.reduce_axis((0, K), name='k')

    lambda_dict = { 
            1:lambda p,n,m: te.sum(A[k,m] * B[p,n,k], axis=k),
            2:lambda p,n,m: te.sum(A[k,m] * B[n,p,k], axis=k),
            3:lambda p,n,m: te.sum(A[k,m] * B[p,k,n], axis=k),
            4:lambda p,n,m: te.sum(A[k,m] * B[n,k,p], axis=k),
            5:lambda p,n,m: te.sum(A[k,m] * B[k,p,n], axis=k),
            6:lambda p,n,m: te.sum(A[k,m] * B[k,n,p], axis=k),

            7:lambda p,n,m: te.sum(A[m,k] * B[p,n,k], axis=k),
            8:lambda p,n,m: te.sum(A[m,k] * B[n,p,k], axis=k),
            9:lambda p,n,m: te.sum(A[m,k] * B[p,k,n], axis=k),
            10:lambda p,n,m: te.sum(A[m,k] * B[n,k,p], axis=k),
            11:lambda p,n,m: te.sum(A[m,k] * B[k,p,n], axis=k),
            12:lambda p,n,m: te.sum(A[m,k] * B[k,n,p], axis=k),

            13:lambda p,n,m: te.sum(A[k,n] * B[p,m,k], axis=k),
            14:lambda p,n,m: te.sum(A[k,n] * B[m,p,k], axis=k),
            15:lambda p,n,m: te.sum(A[k,n] * B[p,k,m], axis=k),
            16:lambda p,n,m: te.sum(A[k,n] * B[m,k,p], axis=k),
            17:lambda p,n,m: te.sum(A[k,n] * B[k,p,m], axis=k),
            18:lambda p,n,m: te.sum(A[k,n] * B[k,m,p], axis=k),

            19:lambda p,n,m: te.sum(A[n,k] * B[p,m,k], axis=k),
            20:lambda p,n,m: te.sum(A[n,k] * B[m,p,k], axis=k),
            21:lambda p,n,m: te.sum(A[n,k] * B[p,k,m], axis=k),
            22:lambda p,n,m: te.sum(A[n,k] * B[m,k,p], axis=k),
            23:lambda p,n,m: te.sum(A[n,k] * B[k,p,m], axis=k),
            24:lambda p,n,m: te.sum(A[n,k] * B[k,m,p], axis=k),

            25:lambda p,n,m: te.sum(A[k,p] * B[n,m,k], axis=k),
            26:lambda p,n,m: te.sum(A[k,p] * B[m,n,k], axis=k),
            27:lambda p,n,m: te.sum(A[k,p] * B[n,k,m], axis=k),
            28:lambda p,n,m: te.sum(A[k,p] * B[m,k,n], axis=k),
            29:lambda p,n,m: te.sum(A[k,p] * B[k,n,m], axis=k),
            30:lambda p,n,m: te.sum(A[k,p] * B[k,m,n], axis=k),

            31:lambda p,n,m: te.sum(A[p,k] * B[n,m,k], axis=k),
            32:lambda p,n,m: te.sum(A[p,k] * B[m,n,k], axis=k),
            33:lambda p,n,m: te.sum(A[p,k] * B[n,k,m], axis=k),
            34:lambda p,n,m: te.sum(A[p,k] * B[m,k,n], axis=k),
            35:lambda p,n,m: te.sum(A[p,k] * B[k,n,m], axis=k),
            36:lambda p,n,m: te.sum(A[p,k] * B[k,m,n], axis=k),
            }

    return tc_template(M,N,P,K, A,B,k, lambda_dict[tc_index], tc_index, dtype)

def tc_template(M,N,P,K, A,B,k,lambda_func, tc_index, dtype):
    C = te.compute((P,N,M), lambda_func, name='C')
    s = te.create_schedule(C.op)

    # schedule
    p,n,m = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    mo, mi = cfg.define_split("tile_m", m, num_outputs=2, policy='verbose')
    no, ni = cfg.define_split("tile_n", n, num_outputs=2, policy='verbose')
    po, pi = cfg.define_split("tile_p", p, num_outputs=2, policy='verbose') #filter=lambda y: y.size[-1] % 8 == 0, policy='verbose')
    ko, ki = cfg.define_split("tile_k", k, num_outputs=2, policy='verbose')

    order = [mo, no, po, ko, mi, ni, pi, ki] 
    perms = []
    #for start in permutations([mo, no, po, ko]):
    start = [mo, no, po, ko]
    for end in permutations([mi, ni, pi, ki]):
        perms.append(list(start)+list(end))

    cfg.define_reorder('reorder_0', order,
            policy='candidate', candidate=perms)
    ##### define space end #####

    # schedule according to config
    mo, mi = cfg["tile_m"].apply(s, C, m)
    no, ni = cfg["tile_n"].apply(s, C, n)
    po, pi = cfg["tile_p"].apply(s, C, p)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    order = [mo, no, po, ko, mi, ni, pi, ki] 
    cfg['reorder_0'].apply(s, C, order)

    #to_parallel = []
    #for i in cfg['reorder_0'].perm:
    #    if i == 3:
    #        break
    #    to_parallel.append(order[i])
    #parallel_axis = s[C].fuse(*to_parallel)

    parallel_axis = s[C].fuse(mo,no,po)
    s[C].parallel(parallel_axis)

    _, cfg.extents, cfg.array_dims, _, cfg.conv_dims, cfg.fastest_varying, cfg.arrays = get_tc_extents_info(M,N,P,K,cfg, tc_index)

    return s, [A, B, C]
