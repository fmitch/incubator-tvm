from data_volume_estimator import estimate_dv

loop_order_a=[0,1,2,3,4,5,6,7,8,9]
loop_order_b=[0,1,2,3,4,5,6,9,7,8]
num_iters_a = [4,8,56,14,64,3,3,1,4,16]
num_iters_b = [4,1,14, 4,64,3,3,4,14,128]
inds = [[0,2,3,4,5,6,7,8], [1,4,5,6,9], [0,1,2,3,7,8,9]]
cache_sizes = [32768 // 4, 1048576 // 4, 23068672 // 4]
conv_inds = [ [ (2,7), (5,)], [(3,8), (6,)]]

d_foot_a, d_vol_a = estimate_dv(loop_order_a, num_iters_a, inds, cache_sizes, conv_inds)
d_foot_b, d_vol_b = estimate_dv(loop_order_b, num_iters_b, inds, cache_sizes, conv_inds)
d_foot_c, d_vol_c = estimate_dv(loop_order_a, num_iters_b, inds, cache_sizes, conv_inds)
d_foot_d, d_vol_d = estimate_dv(loop_order_b, num_iters_a, inds, cache_sizes, conv_inds)

print(d_vol_a[:,:,-1].sum(axis=0))
print(d_vol_b[:,:,-1].sum(axis=0))
print(d_vol_c[:,:,-1].sum(axis=0))
print(d_vol_d[:,:,-1].sum(axis=0))
from IPython import embed
#embed()
