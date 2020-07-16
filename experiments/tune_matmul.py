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
"""
Writing tunable template and Using auto-tuner
=============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

This is an introduction tutorial to the auto-tuning module in TVM.

There are two steps in auto-tuning.
The first step is defining a search space.
The second step is running a search algorithm to explore through this space.
In this tutorial, you can learn how to perform these two steps in TVM.
The whole workflow is illustrated by a matrix multiplication example.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in TVM, we need to install some extra dependencies.
# This step (installing xgboost) can be skipped as it doesn't need XGBoost
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import logging
import sys

import numpy as np
import tvm
from tvm import te

# the module is called `autotvm`
from tvm import autotvm

######################################################################
# Step 1:  Define the search space
# --------------------------------
# In this section, we will rewrite a deterministic TVM schedule code to a
# tunable schedule template. You can regard the process of search space definition
# as the parameterization of our existing schedule code.
#
# To begin with, here is how we implement a blocked matrix multiplication in TVM.

# Matmul V0: Constant tiling factor
def matmul_v0(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

#####################################################################
# Parametrize the schedule
# ^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous schedule code, we use a constant "8" as tiling factor.
# However, it might not be the best one because the best tiling factor depends
# on real hardware environment and input shape.
#
# If you want the schedule code to be portable across a wider range of input shapes
# and target hardware, it is better to define a set of candidate values and
# pick the best one according to the measurement results on target hardware.
#
# In autotvm, we can define a tunable parameter, or a "knob" for such kind of value.

# Matmul V1: List candidate values
@autotvm.template("tutorial/matmul_v1")  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

###############################################################################
# Here we make four modifications to the previous schedule code and get
# a tunable "template". We can explain the modifications one by one.
#
# 1. Use a decorator to mark this function as a simple template.
# 2. Get a config object:
#    You can regard this :code:`cfg` as an argument of this function but
#    we obtain it in a different way. With this argument, this function is no longer
#    a deterministic schedule code. Instead, we can pass different configurations to
#    this function and get different schedules, so this function is a "template".
#
#    To make the template function more compact, we do two things in a single function.
#    (1) define a search space and (2) schedule according to an entity in this space.
#    To achieve this, we make :code:`cfg` be either
#    a :any:`ConfigSpace` or a :any:`ConfigEntity` object.
#
#    When it is a :any:`ConfigSpace`, it will collect all tunable knobs in this function and
#    build the search space.
#    When it is a :any:`ConfigEntity`, it will ignore all space definition API
#    (namely, :code:`cfg.define_XXXXX(...)`).   Instead, it stores deterministic values for
#    all tunable knobs, and we schedule according to these values.
#
#    During auto-tuning, we will first call this template with a :any:`ConfigSpace`
#    object to build the search space. Then we call this template with different :any:`ConfigEntity`
#    in the built space to get different schedules. Finally we will measure the code generated by
#    different schedules and pick the best one.
#
# 3. Define two tunable knobs. The first one is :code:`tile_y` with
#    5 possible values. The second one is :code:`tile_x` with a same
#    list of possible values. These two knobs are independent, so they
#    span a search space with size = 5x5 = 25
# 4. Schedule according to the deterministic values in :code:`cfg`
#

#####################################################################
# Use better space definition API
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous template, we manually list all possible values for a knob.
# This is the lowest level API to define the space.
# However, we also provide another set of API to make the space definition
# easier and smarter. It is recommended to use this set of high level API.
#
# In the following example, we use :any:`ConfigSpace.define_split` to define a split
# knob. It will enumerate all the possible ways to split an axis and construct
# the space.
#
# We also have :any:`ConfigSpace.define_reorder` for reorder knob and
# :any:`ConfigSpace.define_annotate` for annotation like unroll, vectorization,
# thread binding.
# When the high level API cannot meet your requirement, you can always fall
# back to use low level API.

@autotvm.template("tutorial/matmul")
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
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

N, L, M = 512, 512, 512
task = autotvm.task.create("tutorial/matmul", args=(N, L, M, 'float32'), target='llvm')
print(task.config_space)

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("llvm"):
        s, arg_bufs = matmul(N, L, M, 'float32')
        func = tvm.build(s, arg_bufs)

a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
