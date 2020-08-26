import numpy as np

def estimate_dv(loop_order, num_iters, inds, cache_sizes, conv_inds, fastest_varying, arrays=['In_conv', 'K', 'O'], word_size=4, line_size=64, use_full_footprint=True):
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
        Size of each cache level, in bytes, from lowest to highest: [L1, L2, L3, ...]
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
    cache_sizes = np.array(cache_sizes) / line_size
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
    extra_lines = np.zeros(len(arrays))
    d_foot_lower = np.ones((len(arrays), len(loop_order)))
    d_foot_prob = np.ones((len(arrays), len(loop_order)))
    d_foot_upper = np.ones((len(arrays), len(loop_order)))
    d_vol_lower = np.ones((len(arrays), len(cache_sizes), len(loop_order)))
    d_vol_prob = np.ones((len(arrays), len(cache_sizes), len(loop_order)))
    d_vol_upper = np.ones((len(arrays), len(cache_sizes), len(loop_order)))
    ranges = dict.fromkeys(['h','w','kh','kw'], 1)
    loop_order = list(reversed(loop_order))
    num_iters = list(reversed(num_iters))
    L = line_size/word_size

    for loop_ind, loop in enumerate(loop_order):
        if loop == 'op': continue
        for arr_ind, arr in enumerate(arrays):
            if loop in inds[arr_ind] and arr != 'In_conv':
                if loop == inds[arr_ind][-1]:
                    if num_iters[loop_ind] % L == 0:
                        extra_lines[arr_ind] = np.ceil(num_iters[loop_ind]/L) + (L - 1) / L
                    else:
                        extra_lines[arr_ind] = np.ceil(num_iters[loop_ind]/L) + ((num_iters[loop_ind] % L) - 1)/L
                    d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L))
                    d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) + 1)
                    d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1] * extra_lines[arr_ind]
                    for level, cache_size in enumerate(cache_sizes):
                        lower_foot_sum = d_foot_lower[arr_ind, loop_ind]
                        upper_foot_sum = d_foot_upper[arr_ind, loop_ind]
                        prob_foot_sum = d_foot_prob[arr_ind, loop_ind]
                        if use_full_footprint:
                            for arr_ind_other, _ in enumerate(arrays):
                                if arr_ind != arr_ind_other:
                                    lower_foot_sum += d_foot_lower[arr_ind_other, loop_ind-1]
                                    upper_foot_sum += d_foot_upper[arr_ind_other, loop_ind-1]
                                    prob_foot_sum += d_foot_prob[arr_ind_other, loop_ind-1]
                        if prob_foot_sum < cache_size:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) )
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) + 1 )
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1] * extra_lines[arr_ind]
                        else:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1] * num_iters[loop_ind]
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1] * num_iters[loop_ind]
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1] * num_iters[loop_ind]
                elif loop in fastest_varying[arr_ind]:
                    small_loop = num_iters[loop_order.index(inds[arr_ind][-1])]
                    if small_loop <= L/2:
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]*small_loop/L
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]*(1+small_loop)/L
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]*extra_lines[arr_ind]*small_loop/L
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*small_loop/L
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*(1+small_loop)/L
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*(extra_lines[arr_ind] % 1 + small_loop)/L
                    else:
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]% 1
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]%1

                else:
                    d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]
                    d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]
                    d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]
                    for level, _ in enumerate(cache_sizes):
                        d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
            elif loop not in inds[arr_ind]:
                d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]
                d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]
                d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]
        arr_ind = 0
        if arrays[arr_ind] == 'In_conv':
            if loop in inds[arr_ind]:
                if loop not in sum(sum(conv_inds, []),()):
                    if loop == inds[arr_ind][-1]:
                        if num_iters[loop_ind] % L == 0:
                            extra_lines[arr_ind] = np.ceil(num_iters[loop_ind]/L) + (L - 1) / L
                        else:
                            extra_lines[arr_ind] = np.ceil(num_iters[loop_ind]/L) + ((num_iters[loop_ind] % L) - 1)/L
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L))
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) + 1)
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1] * extra_lines[arr_ind]
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) )
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1] * (np.ceil(num_iters[loop_ind]/L) + 1 )
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1] * extra_lines[arr_ind]

                    elif loop in fastest_varying[arr_ind]:
                        small_loop = num_iters[loop_order.index(inds[arr_ind][-1])]
                        if small_loop <= L/2:
                            d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]*small_loop/L
                            d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]*(1+small_loop)/L
                            d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]*extra_lines[arr_ind]*small_loop/L
                            for level, _ in enumerate(cache_sizes):
                                d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*small_loop/L
                                d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*(1+small_loop)/L
                                d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]*(extra_lines[arr_ind] % 1 + small_loop)/L
                        else:
                            d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]
                            d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]
                            d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]% 1
                            for level, _ in enumerate(cache_sizes):
                                d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                                d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                                d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]%1

                    else:
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*num_iters[loop_ind]
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*num_iters[loop_ind]
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*num_iters[loop_ind]
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                elif loop in sum(sum(conv_inds, []),()):
                    counter, key = conv_keys[loop]
                    other_key = (1+key) % 2
                    new_range = conv_ranges[counter,key]*num_iters[loop_ind]
                    mul_factor = (new_range + conv_ranges[counter,other_key]-1)/(conv_ranges[counter,key] + conv_ranges[counter,other_key]-1)
                    if loop == inds[arr_ind][-1]:
                        if mul_factor % L < 1:
                            extra_lines[arr_ind] = np.ceil(mul_factor/L) + (L - 1) / L
                        else:
                            extra_lines[arr_ind] = np.ceil(mul_factor/L) + ((mul_factor % L) - 1)/L
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1] * (np.ceil(mul_factor/L) )
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1] * (np.ceil(mul_factor/L) + 1 )
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1] * extra_lines[arr_ind]
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = (np.ceil(mul_factor/L) )
                            d_vol_upper[arr_ind, level, loop_ind] = (np.ceil(mul_factor/L) + 1 )
                            d_vol_prob[arr_ind, level, loop_ind] =  extra_lines[arr_ind]




                    elif loop in fastest_varying[arr_ind]:
                        small_loop = num_iters[loop_order.index(inds[arr_ind][-1])] + 2
                        if small_loop <= L/2:
                            d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*mul_factor*small_loop/L
                            d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*mul_factor*(1+small_loop)/L
                            d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor*extra_lines[arr_ind]*small_loop/L
                            for level, _ in enumerate(cache_sizes):
                                d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*mul_factor*small_loop/L
                                d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*mul_factor*(1+small_loop)/L
                                d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor*(extra_lines[arr_ind] % 1 + small_loop)/L
                        else:
                            d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*mul_factor
                            d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*mul_factor
                            d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]% 1
                            for level, _ in enumerate(cache_sizes):
                                d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*mul_factor
                                d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*mul_factor
                                d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]%1


                    else:
                        d_foot_lower[arr_ind, loop_ind] = d_foot_lower[arr_ind, loop_ind-1]*mul_factor
                        d_foot_upper[arr_ind, loop_ind] = d_foot_upper[arr_ind, loop_ind-1]*mul_factor
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor
                        for level, _ in enumerate(cache_sizes):
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*mul_factor
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*mul_factor
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor
                    conv_ranges[counter,key] = new_range

        for arr_ind, arr in enumerate(arrays):
            if loop not in inds[arr_ind]:
                for level, cache_size in enumerate(cache_sizes):
                    if use_full_footprint:
                        lower_foot_sum = d_foot_lower[arr_ind, loop_ind]
                        upper_foot_sum = d_foot_upper[arr_ind, loop_ind]
                        prob_foot_sum = d_foot_prob[arr_ind, loop_ind]
                        for arr_ind_other, _ in enumerate(arrays):
                            if arr_ind != arr_ind_other:
                                lower_foot_sum += d_foot_lower[arr_ind_other, loop_ind-1]
                                upper_foot_sum += d_foot_upper[arr_ind_other, loop_ind-1]
                                prob_foot_sum += d_foot_prob[arr_ind_other, loop_ind-1]
                        if lower_foot_sum < cache_size:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        if upper_foot_sum < cache_size:
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        if prob_foot_sum < cache_size:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                    else:
                        if d_foot_lower[arr_ind,loop_ind] < cache_size:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_lower[arr_ind, level, loop_ind] = d_vol_lower[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        if d_foot_upper[arr_ind,loop_ind] < cache_size:
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_upper[arr_ind, level, loop_ind] = d_vol_upper[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                        if d_foot_prob[arr_ind,loop_ind] < cache_size:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]

    return (d_foot_lower.astype(np.int64), d_foot_upper.astype(np.int64), d_foot_prob.astype(np.int64)), (d_vol_lower.astype(np.int64), d_vol_upper.astype(np.int64), d_vol_prob.astype(np.int64))


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


def estimate_dv2(loop_order, num_iters, inds, cache_sizes, conv_inds, fastest_varying, arrays=['In_conv', 'K', 'O'], word_size=4, line_size=64, use_full_footprint=True):
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
        Size of each cache level, in bytes, from lowest to highest: [L1, L2, L3, ...]
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
    cache_sizes = np.array(cache_sizes) / line_size
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
    extra_lines = np.zeros(len(arrays))
    d_foot_prob = np.ones((len(arrays), len(loop_order)))
    d_vol_prob = np.ones((len(arrays), len(cache_sizes), len(loop_order)))
    ranges = dict.fromkeys(['h','w','kh','kw'], 1)
    loop_order = list(reversed(loop_order))
    num_iters = list(reversed(num_iters))
    L = line_size/word_size

    for loop_ind, loop in enumerate(loop_order):
        if loop == 'op': continue
        for arr_ind, arr in enumerate(arrays):
            if loop in inds[arr_ind]:
                if arr != 'In_conv' or loop not in sum(sum(conv_inds, []),()):
                    mul_factor = num_iters[loop_ind]
                else:
                    counter, key = conv_keys[loop]
                    other_key = (1+key) % 2
                    new_range = conv_ranges[counter, key]*num_iters[loop_ind]
                    mul_factor = (new_range + conv_ranges[counter, other_key]-1)/(conv_ranges[counter, key] + conv_ranges[counter, other_key]-1)
                    conv_ranges[counter,key] = new_range

                if loop == inds[arr_ind][-1]:
                    if mul_factor % L < 1:
                        extra_lines[arr_ind] = np.ceil(mul_factor/L) + (L - 1) / L
                    else:
                        extra_lines[arr_ind] = np.ceil(mul_factor/L) + ((mul_factor % L) - 1)/L
                    d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1] * extra_lines[arr_ind]
                    for level, _ in enumerate(cache_sizes):
                        d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1] * extra_lines[arr_ind]
                elif loop in fastest_varying[arr_ind]:
                    small_loop = num_iters[loop_order.index(inds[arr_ind][-1])]
                    if small_loop <= L/2:
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor*extra_lines[arr_ind]*small_loop/L
                        for level, _ in enumerate(cache_sizes):
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor*(extra_lines[arr_ind] % 1 + small_loop)/L
                    else:
                        d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]% 1
                        for level, _ in enumerate(cache_sizes):
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor/extra_lines[arr_ind] * int(extra_lines[arr_ind]) + extra_lines[arr_ind]%1
                else:
                    d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]*mul_factor
                    for level, _ in enumerate(cache_sizes):
                        d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*mul_factor
            elif loop not in inds[arr_ind]:
                d_foot_prob[arr_ind, loop_ind] = d_foot_prob[arr_ind, loop_ind-1]

        for arr_ind, arr in enumerate(arrays):
            if loop not in inds[arr_ind]:
                for level, cache_size in enumerate(cache_sizes):
                    if use_full_footprint:
                        prob_foot_sum = d_foot_prob[arr_ind, loop_ind]
                        for arr_ind_other, _ in enumerate(arrays):
                            if arr_ind != arr_ind_other:
                                prob_foot_sum += d_foot_prob[arr_ind_other, loop_ind-1]
                        if prob_foot_sum < cache_size:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]
                    else:
                        if d_foot_prob[arr_ind,loop_ind] < cache_size:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]
                        else:
                            d_vol_prob[arr_ind, level, loop_ind] = d_vol_prob[arr_ind, level, loop_ind-1]*num_iters[loop_ind]

    return d_foot_prob.astype(np.int64), d_vol_prob.astype(np.int64)



















