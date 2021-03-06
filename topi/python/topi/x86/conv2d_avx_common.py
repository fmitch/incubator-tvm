# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""Conv2D schedule on for Intel CPU"""
import tvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from ..generic import conv2d as conv2d_generic
from ..util import get_const_tuple
from .tensor_intrin import dot_16x1x16_uint8_int8_int32
from .util import get_fp32_len

def _fallback_schedule(cfg, wkl):
    simd_width = get_fp32_len()
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if wkl.out_filter % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _fallback_schedule_int8(cfg, wkl):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 16
    assert wkl.out_filter % oc_bn == 0

    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break
    assert wkl.in_filter % 4 == 0

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _schedule_conv_NCHW_dv(s, cfg, data_vec, kernel_vec, conv_out, last):
    n, co, oh, ow  = s[conv_out].op.axis
    ci, kh, kw = s[conv_out].op.reduce_axis
    vci_n = cfg['tile_ic'].size[-1]
    vco_n = cfg['tile_oc'].size[-1]
    vh_n = cfg['tile_oh'].size[-1]
    vw_n = cfg['tile_ow'].size[-1]
    co_n = co.dom.extent.value 
    oh_n = oh.dom.extent.value // vh_n
    ow_n = ow.dom.extent.value // vw_n
    ci_n = ci.dom.extent.value // vci_n
    n_n = n.dom.extent.value
    kh_n = kh.dom.extent.value
    kw_n = kw.dom.extent.value
    cfg.extents = [n_n, co_n, oh_n, ow_n, ci_n, vci_n, kh_n, kw_n, vh_n, vw_n, vco_n]
    cfg.arrays = None
    order = cfg['reorder_0'].perm #[n, co, oh, ow, ci, vci, kh, kw, vh, vw, vco]
    cfg.array_dims = [ [0,2,3,4,6,7,8,9,5], [1,4,5,6,7,10], [0,1,2,3,8,9,10] ]
    cfg.conv_dims = [ [(2,8), (6,)], [(3,9), (7,)] ]
    cfg.fastest_varying = [ [8], [6], [8]]

    # schedule conv
    ci, vci = s[conv_out].split(ci, factor=vci_n)
    co, vco = s[conv_out].split(co, factor=vco_n)
    oh, vh = s[conv_out].split(oh, factor=vh_n)
    ow, vw = s[conv_out].split(ow, factor=vw_n)
    order = [n, co, oh, ow, ci, vci, kh, kw, vh, vw, vco]
    cfg['reorder_0'].apply(s, conv_out, order)

    # schedule fusion
    #reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    #if unroll_kw:
    #    s[last].unroll(kw)

    # mark parallel

    s[conv_out].vectorize(vco)
    s[conv_out].unroll(vw)

    parallel_axis = s[conv_out].fuse(n, co, oh)
    s[last].parallel(parallel_axis)
    return s


def _schedule_conv_NCHWc_dv(s, cfg, data_vec, kernel_vec, conv_out, last):
    n, co, oh, ow, vco = s[conv_out].op.axis
    ci, kh, kw = s[conv_out].op.reduce_axis
    vci_n = cfg['tile_ic'].size[-1]
    vco_n = cfg['tile_oc'].size[-1]
    vh_n = cfg['tile_oh'].size[-1]
    vw_n = cfg['tile_ow'].size[-1]
    assert( vco_n == vco.dom.extent.value)
    co_n = co.dom.extent.value 
    oh_n = oh.dom.extent.value // vh_n
    ow_n = ow.dom.extent.value // vw_n
    ci_n = ci.dom.extent.value // vci_n
    n_n = n.dom.extent.value
    kh_n = kh.dom.extent.value
    kw_n = kw.dom.extent.value
    cfg.extents = [n_n, co_n, oh_n, ow_n, ci_n, vci_n, kh_n, kw_n, vh_n, vw_n, vco_n]
    cfg.arrays = None
    order = cfg['reorder_0'].perm #[n, co, oh, ow, ci, vci, kh, kw, vh, vw, vco]
    #cfg.array_dims = [ [0,2,3,4,6,7,8,9,5], [1,4,5,6,7,10], [0,1,2,3,8,9,10] ]
    #cfg.conv_dims = [ [(2,8), (6,)], [(3,9), (7,)] ]
    #cfg.fastest_varying = [ [5], [10], [10]]
    cfg.array_dims = [ [1,3,4,5,7,8,9,10,6], [2,5,6,7,8,11], [1,2,3,4,9,10,11] ]
    cfg.conv_dims = [ [(3,9), (7,)], [(4,10), (8,)] ]
    cfg.fastest_varying = [ [6], [11], [11]]


    if isinstance(s[data_vec].op, tvm.te.ComputeOp) \
            and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        s[data_vec].vectorize(ic_block)
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and \
            kernel_vec.name == 'kernel_vec':
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, ic_block, iw = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        oc_chunk, ic_chunk, oh_k, ow_k, ic_block, oc_block = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh_k, ic_chunk, ow_k, ic_block, oc_block)
        oc_bn = cfg["tile_oc"].size[-1]
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh_k)
        s[kernel_vec].parallel(parallel_axis)


    
    # schedule conv
    ci, vci = s[conv_out].split(ci, factor=vci_n)
    #co, vco = s[conv_out].split(co, factor=vco_n)
    oh, vh = s[conv_out].split(oh, factor=vh_n)
    ow, vw = s[conv_out].split(ow, factor=vw_n)
    order = [n, co, oh, ow, ci, vci, kh, kw, vh, vw, vco]
    cfg['reorder_0'].apply(s, conv_out, order)

    # schedule fusion
    #reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    #if unroll_kw:
    #    s[last].unroll(kw)

    # mark parallel

    #s[conv_out].vectorize(vco)
    #s[conv_out].unroll(vw)

    parallel_axis = s[conv_out].fuse(n, co, oh)
    s[last].parallel(parallel_axis)
    return s


def _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, last):
    # fetch schedule
    reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)

    # schedule pad
    if isinstance(s[data_vec].op, tvm.te.ComputeOp) \
            and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        s[data_vec].vectorize(ic_block)
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and \
            kernel_vec.name == 'kernel_vec':
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, ic_block, iw = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
        oc_bn = cfg["tile_oc"].size[-1]
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh)
        s[kernel_vec].parallel(parallel_axis)


    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
            parallel_axis = s[O].fuse(batch, oc_chunk, oh)
            s[C].compute_at(s[O], parallel_axis)
            s[O].vectorize(oc_block)
            s[O].parallel(parallel_axis)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)

    return s


def _schedule_conv_NCHWc_int8(s, cfg, data_vec, kernel_vec, conv_out, last):
    return conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(s, cfg, data_vec, kernel_vec,
                                                              conv_out, last, int32_lanes=16,
                                                              intrin=dot_16x1x16_uint8_int8_int32())
