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
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging

import numpy as np

from ..measure import MeasureInput, create_measure_batch
from ..util import format_si_prefix

from ..env import GLOBAL_SCOPE
from .data_volume_estimator import estimate_dv

import subprocess

logger = logging.getLogger('autotvm')
import tvm
import pylikwid
import time

cache_sizes = [
        int(subprocess.run(['getconf', 'LEVEL1_DCACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL2_CACHE_SIZE'], stdout=subprocess.PIPE).stdout),
        int(subprocess.run(['getconf', 'LEVEL3_CACHE_SIZE'], stdout=subprocess.PIPE).stdout)]

class SavedFeature:
    def __init__(self, config=None, feature=None, result=None, counters=None):
        self.config = config
        self.feature = feature
        self.result = result
        self.counters = counters

    def set_result(self, result):
        self.result = result

    def set_config(self, config):
        self.config = config

    def set_feature(self, feature):
        self.feature = feature

    def set_counters(self, counters):
        self.counters = counters


class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task """ 
    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """


    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix='G', likwid_event=None, save_features=False, random=False):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        si_prefix: str
            One of tvm.autotvm.util.SI_PREFIXES. The SI prefix to use when reporting FLOPS.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping
        start_time = time.time()

        # Validate si_prefix arg
        format_si_prefix(0, si_prefix)

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0

        if likwid_event != None:
            # Get arrays for conv
            N, CI, H, W = self.task.args[0][1]
            CO, _, KH, KW = self.task.args[1][1]
            padding = self.task.args[3]

        ctx=tvm.context(self.task.target.__str__(), 0)
        #a_tvm = tvm.nd.array(np.random.uniform(size=(N,CI,H,W) ).astype(np.float32), ctx)
        #w_tvm = tvm.nd.array(np.random.uniform(size=(CO,CI,KH,KW) ).astype(np.float32), ctx)
        #c_tvm = tvm.nd.array(np.zeros((N,CO,H+KH-2*padding-1,W+KW-2*padding-1), dtype=np.float32), ctx)

        while i < n_trial:
            if not self.has_next():
                break

            if random:
                configs = self.random_next_batch(min(n_parallel, n_trial - i))
            else:
                configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug("No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                             i + k + 1, si_prefix, format_si_prefix(flops, si_prefix),
                             format_si_prefix(self.best_flops, si_prefix), res, config)
                
            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            if random:
                self.update_random(inputs, results)
            else:
                self.update(inputs, results)

            if likwid_event != None:
                pylikwid.inittopology()
                cpu_topo = pylikwid.getcputopology()
                cpus = list(range(cpu_topo['activeHWThreads']))
                pylikwid.finalizetopology()

                err = pylikwid.init(cpus)
                group = pylikwid.addeventset(likwid_event)
                err = pylikwid.setup(group)

                for k, (inp, res) in enumerate(zip(inputs, results)):
                    with inp.target:
                        sch, args = self.task.instantiate(inp.config)
                        #with tvm.ir.transform.PassContext():
                        func = tvm.build(sch, args, target_host=inp.task.target_host)
                        evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3, number=4)

                    dshape = (N,CI//inp.config['tile_ic'].size[-1],H,W,inp.config['tile_ic'].size[-1])
                    kshape = (CO//inp.config['tile_oc'].size[-1],CI//inp.config['tile_ic'].size[-1],KH,KW,inp.config['tile_ic'].size[-1],inp.config['tile_oc'].size[-1])
                    oshape = (N,CO//inp.config['tile_oc'].size[-1],H+KH-2*padding-1,W+KW-2*padding-1, inp.config['tile_oc'].size[-1])
                    a_tvm = tvm.nd.array(np.random.uniform(size=dshape).astype(np.float32), ctx)
                    w_tvm = tvm.nd.array(np.random.uniform(size=kshape).astype(np.float32), ctx)
                    c_tvm = tvm.nd.array(np.zeros(oshape, dtype=np.float32), ctx)
                    ##Warm up ### I tried this warm up and running the function once, 
                    #             likwid results were very bad, resulted in barely better than
                    #             random when training RandForest model on post-tuning data
                    #if tuple(args[1].shape) == w_tvm.shape:
                    #    for _ in range(10):
                    #        func(c_tvm, w_tvm, a_tvm)
                    #else:
                    #    for _ in range(10):
                    #        func(c_tvm, a_tvm, w_tvm)

                    #LIKWID PERFCTR
                    err = pylikwid.start()
                    if tuple(args[1].shape) == w_tvm.shape:
                        evaluator(c_tvm, w_tvm, a_tvm)
                    else:
                        evaluator(c_tvm, a_tvm, w_tvm)
                    err = pylikwid.stop()

                    likwid_results = []
                    for thread in range(0,len(cpus)):
                        likwid_results.append({})
                        for event_num in range(pylikwid.getnumberofevents(group)):
                            key = pylikwid.getnameofevent(group, event_num)
                            if key in likwid_results[-1].keys():
                                likwid_results[-1][key] += pylikwid.getresult(group,event_num, thread)
                            else:
                                likwid_results[-1][key] = pylikwid.getresult(group,event_num, thread)
                    #END LIKWID PERFCTR

                    if inp.config.index in self.cost_model.saved_features.keys():
                        self.cost_model.saved_features[inp.config.index].set_result(res)
                        self.cost_model.saved_features[inp.config.index].set_counters(likwid_results)
                    else:
                        self.cost_model.saved_features[inp.config.index] = SavedFeature(result=res, counters=likwid_results)
                pylikwid.finalize()
            elif save_features == True:
                for k, (inp, res) in enumerate(zip(inputs, results)):
                    if inp.config.index in self.cost_model.saved_features.keys():
                        self.cost_model.saved_features[inp.config.index].set_result(res)
                    else:
                        self.cost_model.saved_features[inp.config.index] = SavedFeature(result=res)
            if len(self.cost_model.saved_features['scores']) > 0:
                self.cost_model.saved_features['scores'][-1].append(time.time() - start_time)

            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
