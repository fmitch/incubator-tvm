from data_volume_estimator import *
import numpy as np

loop_order_a=[0,1,2,3,4,5,6,7,8,9]
loop_order_b=[0,1,2,3,4,5,6,9,7,8]
num_iters_a = [4,1,56,14,64,3,3,1,4,128]
num_iters_c = [4,2,56,14,64,3,3,1,4,64]
num_iters_e = [4,4,56,14,64,3,3,1,4,32]
num_iters_g = [4,8,56,14,64,3,3,1,4,16]
#num_iters_b = [4,1,14, 4,64,3,3,4,14,128]
#num_iters_d = [4,2,14, 4,64,3,3,4,14,64]
#num_iters_f = [4,4,14, 4,64,3,3,4,14,32]
#num_iters_h = [4,8,14, 4,64,3,3,4,14,16]
inds = [[0,2,3,4,5,6,7,8], [5,6,4,1,9], [0,1,2,3,7,9,8]]
cache_sizes = np.array([32*1024 , 1024*1024 , 27.5*1024*1024]) // 64
conv_inds = [ [ (2,7), (5,)], [(3,8), (6,)]]

d_foot_a1, d_vol_a1 = estimate_dv(loop_order_a, num_iters_a, inds, cache_sizes, conv_inds)
d_foot_a2, d_vol_a2 = estimate_dv(loop_order_a, num_iters_a, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_b1, d_vol_b1 = estimate_dv(loop_order_b, num_iters_a, inds, cache_sizes, conv_inds)
d_foot_b2, d_vol_b2 = estimate_dv(loop_order_b, num_iters_a, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_c1, d_vol_c1 = estimate_dv(loop_order_a, num_iters_c, inds, cache_sizes, conv_inds)
d_foot_c2, d_vol_c2 = estimate_dv(loop_order_a, num_iters_c, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_d1, d_vol_d1 = estimate_dv(loop_order_b, num_iters_c, inds, cache_sizes, conv_inds)
d_foot_d2, d_vol_d2 = estimate_dv(loop_order_b, num_iters_c, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_e1, d_vol_e1 = estimate_dv(loop_order_a, num_iters_e, inds, cache_sizes, conv_inds)
d_foot_e2, d_vol_e2 = estimate_dv(loop_order_a, num_iters_e, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_f1, d_vol_f1 = estimate_dv(loop_order_b, num_iters_e, inds, cache_sizes, conv_inds)
d_foot_f2, d_vol_f2 = estimate_dv(loop_order_b, num_iters_e, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_g1, d_vol_g1 = estimate_dv(loop_order_a, num_iters_g, inds, cache_sizes, conv_inds)
d_foot_g2, d_vol_g2 = estimate_dv(loop_order_a, num_iters_g, inds, cache_sizes, conv_inds,use_full_footprint=False)
d_foot_h1, d_vol_h1 = estimate_dv(loop_order_b, num_iters_g, inds, cache_sizes, conv_inds)
d_foot_h2, d_vol_h2 = estimate_dv(loop_order_b, num_iters_g, inds, cache_sizes, conv_inds,use_full_footprint=False)

print(np.array(cache_sizes))
print('A1 Lowr', d_vol_a1[0][:,:,-1].sum(axis=0), 'A2 Lowr', d_vol_a2[0][:,:,-1].sum(axis=0)) 
print('A1 Uppr', d_vol_a1[1][:,:,-1].sum(axis=0), 'A2 Uppr', d_vol_a2[1][:,:,-1].sum(axis=0)) 
print('A1 Prob', d_vol_a1[2][:,:,-1].sum(axis=0), 'A2 Prob', d_vol_a2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('B1 Lowr', d_vol_b1[0][:,:,-1].sum(axis=0), 'B2 Lowr', d_vol_b2[0][:,:,-1].sum(axis=0)) 
print('B1 Uppr', d_vol_b1[1][:,:,-1].sum(axis=0), 'B2 Uppr', d_vol_b2[1][:,:,-1].sum(axis=0)) 
print('B1 Prob', d_vol_b1[2][:,:,-1].sum(axis=0), 'B2 Prob', d_vol_b2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('C1 Lowr', d_vol_c1[0][:,:,-1].sum(axis=0), 'C2 Lowr', d_vol_c2[0][:,:,-1].sum(axis=0)) 
print('C1 Uppr', d_vol_c1[1][:,:,-1].sum(axis=0), 'C2 Uppr', d_vol_c2[1][:,:,-1].sum(axis=0)) 
print('C1 Prob', d_vol_c1[2][:,:,-1].sum(axis=0), 'C2 Prob', d_vol_c2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('D1 Lowr', d_vol_d1[0][:,:,-1].sum(axis=0), 'D2 Lowr', d_vol_d2[0][:,:,-1].sum(axis=0)) 
print('D1 Uppr', d_vol_d1[1][:,:,-1].sum(axis=0), 'D2 Uppr', d_vol_d2[1][:,:,-1].sum(axis=0)) 
print('D1 Prob', d_vol_d1[2][:,:,-1].sum(axis=0), 'D2 Prob', d_vol_d2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('E1 Lowr', d_vol_e1[0][:,:,-1].sum(axis=0), 'E2 Lowr', d_vol_e2[0][:,:,-1].sum(axis=0)) 
print('E1 Uppr', d_vol_e1[1][:,:,-1].sum(axis=0), 'E2 Uppr', d_vol_e2[1][:,:,-1].sum(axis=0)) 
print('E1 Prob', d_vol_e1[2][:,:,-1].sum(axis=0), 'E2 Prob', d_vol_e2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('F1 Lowr', d_vol_f1[0][:,:,-1].sum(axis=0), 'F2 Lowr', d_vol_f2[0][:,:,-1].sum(axis=0)) 
print('F1 Uppr', d_vol_f1[1][:,:,-1].sum(axis=0), 'F2 Uppr', d_vol_f2[1][:,:,-1].sum(axis=0)) 
print('F1 Prob', d_vol_f1[2][:,:,-1].sum(axis=0), 'F2 Prob', d_vol_f2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('G1 Lowr', d_vol_g1[0][:,:,-1].sum(axis=0), 'G2 Lowr', d_vol_g2[0][:,:,-1].sum(axis=0)) 
print('G1 Uppr', d_vol_g1[1][:,:,-1].sum(axis=0), 'G2 Uppr', d_vol_g2[1][:,:,-1].sum(axis=0)) 
print('G1 Prob', d_vol_g1[2][:,:,-1].sum(axis=0), 'G2 Prob', d_vol_g2[2][:,:,-1].sum(axis=0))
print()                                                                                     
print('H1 Lowr', d_vol_h1[0][:,:,-1].sum(axis=0), 'H2 Lowr', d_vol_h2[0][:,:,-1].sum(axis=0)) 
print('H1 Uppr', d_vol_h1[1][:,:,-1].sum(axis=0), 'H2 Uppr', d_vol_h2[1][:,:,-1].sum(axis=0)) 
print('H1 Prob', d_vol_h1[2][:,:,-1].sum(axis=0), 'H2 Prob', d_vol_h2[2][:,:,-1].sum(axis=0))
print()

#d_foot_a_old, d_vol_a_old = estimate_dv_old(loop_order_a, num_iters_a, inds, cache_sizes, conv_inds)
#d_foot_b_old, d_vol_b_old = estimate_dv_old(loop_order_b, num_iters_b, inds, cache_sizes, conv_inds)
#d_foot_c_old, d_vol_c_old = estimate_dv_old(loop_order_a, num_iters_b, inds, cache_sizes, conv_inds)
#d_foot_d_old, d_vol_d_old = estimate_dv_old(loop_order_b, num_iters_a, inds, cache_sizes, conv_inds)
#
#print(np.array(cache_sizes))
#print('A', d_vol_a_old[:,:,-1].sum(axis=0))
#print('B', d_vol_b_old[:,:,-1].sum(axis=0))
#print('C', d_vol_c_old[:,:,-1].sum(axis=0))
#print('D', d_vol_d_old[:,:,-1].sum(axis=0))
from IPython import embed
embed()
