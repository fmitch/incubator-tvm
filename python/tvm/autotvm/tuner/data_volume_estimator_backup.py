import numpy as np

def estimate_dv(loop_order, num_iters, inds, cache_sizes, conv_inds, arrays=['I', 'K', 'O']):
    """ Estimate data volume using loop iterators, arrays, and cache capacities.

    Paramters
    ---------
    loop_order: list()
        The list of loop iterators from outermost to innermost. 
        Must be of same type as inds.
    num_iters: list(int)
        The extent of each loop in loop_order
    inds: list(list())
        List of loop iterators for each array. Outermost to innermost
        In order, Input, Kernel, Output, or same as arrays=['I', 'K', 'O']
        Must be of same type as loop_order.
    cache_sizes: list(ind)
        Size of each cache level, in words, from lowest to highest: [L1, L2, L3, ...]
    conv_inds: list( list(tuple(), tuple()) )
        List of list of two tuples, where each tuple contains keys from loop_order
        for the input and kernel keys corresponding to a dimension of convolution.
        For example, if there were 3-level tiling on H (2,4,9) and 2-level tiling
        on KH (5, 11), then the pair of tuples would be [(2,4,9), (5,11)].
        2D or 3D convolution requires multiple such pairs, such as 
        [ [(2,4,9), (5,11)], [(10), (12)] ]. In this case there is no tiling for
        the second dimension.
        Keys must be the same type as in loop_order and inds
    """
    cache_sizes = np.array(cache_sizes)
    assert(len(inds) == len(arrays))
    counter = 0
    conv_ranges = np.ones((len(conv_inds), 2))
    conv_keys = {}
    for conv_dim in conv_inds:
        for key in conv_dim[0]:
            conv_keys[key] = (counter, 0)
        for key in conv_dim[1]:
            conv_keys[key] = (counter, 1)
        counter += 1 

    num_iters = list(np.array(num_iters)[loop_order]) + [1]
    loop_order = loop_order.copy() + ['op']
    d_foot = np.ones((len(arrays), len(loop_order))).astype(np.uint64)
    d_vol = np.ones((len(arrays), len(cache_sizes), len(loop_order))).astype(np.uint64)
    ranges = dict.fromkeys(['h','w','kh','kw'], 1)
    loop_order = list(reversed(loop_order))
    num_iters = list(reversed(num_iters))

    for loop_ind, loop in enumerate(loop_order):
        if loop == 'op': continue
        for arr_ind, arr in enumerate(arrays):
            if loop in inds[arr_ind] and arr != 'I':
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*num_iters[loop_ind]
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
            elif loop not in inds[arr_ind]:
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]
                #for level, cache_size in enumerate(cache_sizes):
                #    if d_foot[:, loop_ind-1].sum() < cache_size:
                #        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]
                #    else:
                #        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
        arr_ind = 0
        if loop in inds[arr_ind]:
            if loop not in sum(sum(conv_inds, []),()):
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*num_iters[loop_ind]
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
            elif loop in sum(sum(conv_inds, []),()):
                counter, key = conv_keys[loop]
                other_key = (1+key) % 2
                new_range = conv_ranges[counter,key]*num_iters[loop_ind]
                mul_factor = (new_range + conv_ranges[counter,other_key]-1)/(conv_ranges[counter,key] + conv_ranges[counter,other_key]-1)
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*mul_factor
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*mul_factor
                conv_ranges[counter,key] = new_range

        for arr_ind, arr in enumerate(arrays):
            if loop not in inds[arr_ind]:
                for level, cache_size in enumerate(cache_sizes):
                    if d_foot[:, loop_ind].sum() < cache_size:
                        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]
                    else:
                        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]

    return d_foot, d_vol


def estimate_dv_old(loop_order, num_iters, inds, cache_sizes, conv_inds, arrays=['I', 'K', 'O']):
    """ Estimate data volume using loop iterators, arrays, and cache capacities.

    Paramters
    ---------
    loop_order: list()
        The list of loop iterators from outermost to innermost. 
        Must be of same type as inds.
    num_iters: list(int)
        The extent of each loop in loop_order
    inds: list(list())
        List of loop iterators for each array. Outermost to innermost
        In order, Input, Kernel, Output, or same as arrays=['I', 'K', 'O']
        Must be of same type as loop_order.
    cache_sizes: list(ind)
        Size of each cache level, in words, from lowest to highest: [L1, L2, L3, ...]
    conv_inds: list( list(tuple(), tuple()) )
        List of list of two tuples, where each tuple contains keys from loop_order
        for the input and kernel keys corresponding to a dimension of convolution.
        For example, if there were 3-level tiling on H (2,4,9) and 2-level tiling
        on KH (5, 11), then the pair of tuples would be [(2,4,9), (5,11)].
        2D or 3D convolution requires multiple such pairs, such as 
        [ [(2,4,9), (5,11)], [(10), (12)] ]. In this case there is no tiling for
        the second dimension.
        Keys must be the same type as in loop_order and inds
    """
    assert(len(inds) == len(arrays))
    counter = 0
    conv_ranges = np.ones((len(conv_inds), 2))
    conv_keys = {}
    for conv_dim in conv_inds:
        for key in conv_dim[0]:
            conv_keys[key] = (counter, 0)
        for key in conv_dim[1]:
            conv_keys[key] = (counter, 1)
        counter += 1 

    num_iters = list(np.array(num_iters)[loop_order]) + [1]
    loop_order = loop_order.copy() + ['op']
    d_foot = np.ones((len(arrays), len(loop_order))).astype(np.uint64)
    d_vol = np.ones((len(arrays), len(cache_sizes), len(loop_order))).astype(np.uint64)
    ranges = dict.fromkeys(['h','w','kh','kw'], 1)
    loop_order = list(reversed(loop_order))
    num_iters = list(reversed(num_iters))

    for loop_ind, loop in enumerate(loop_order):
        if loop == 'op': continue
        for arr_ind, arr in enumerate(arrays):
            if loop in inds[arr_ind] and arr != 'I':
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*num_iters[loop_ind]
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
            elif loop not in inds[arr_ind]:
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]
                for level, cache_size in enumerate(cache_sizes):
                    if d_foot[arr_ind, loop_ind-1] < cache_size:
                        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]
                    else:
                        d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
        arr_ind = 0
        if loop in inds[arr_ind]:
            if loop not in sum(sum(conv_inds, []),()):
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*num_iters[loop_ind]
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
            elif loop in sum(sum(conv_inds, []),()):
                counter, key = conv_keys[loop]
                other_key = (1+key) % 2
                new_range = conv_ranges[counter,key]*num_iters[loop_ind]
                mul_factor = (new_range + conv_ranges[counter,other_key]-1)/(conv_ranges[counter,key] + conv_ranges[counter,other_key]-1)
                d_foot[arr_ind, loop_ind] = d_foot[arr_ind, loop_ind-1]*mul_factor
                for level, _ in enumerate(cache_sizes):
                    d_vol[arr_ind, level, loop_ind] = d_vol[arr_ind, level, loop_ind-1]*mul_factor
                conv_ranges[counter,key] = new_range

    return d_foot, d_vol
