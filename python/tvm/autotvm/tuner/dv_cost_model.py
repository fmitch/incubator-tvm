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
# pylint: disable=invalid-name
"""XGBoost as cost model"""

import multiprocessing
import logging
import time

import numpy as np

from .. import feature
from ..util import get_rank
from .metric import max_curve, recall_curve, cover_curve
from .model_based_tuner import CostModel, FeatureCache
from .tuner import SavedFeature, cache_sizes
from .data_volume_estimator import estimate_dv
import tvm

logger = logging.getLogger('autotvm')

class DataVolumePredictor():
    def __init__(self, output_type):
        def sum_datavol(datavol):
            assert(len(datavol.shape) == 2)
            return -1*(datavol[:,0]/datavol[:,0].max() + datavol[:,1]/datavol[:,1].max() + datavol[:,2]/datavol[:,2].max() )
        
        def sum_datavolL2L3(datavol):
            assert(len(datavol.shape) == 2)
            return -1*(datavol[:,1]/datavol[:,1].max() + datavol[:,2]/datavol[:,2].max() )
        
        def time_estimate(datavol):
            assert(len(datavol.shape) == 2)
            return -1*(datavol[:,0]*64/100e9 + datavol[:,1]*64/44e9 + datavol[:,2]*64/25e9 )

        if output_type == 'sum':
            self.predict = sum_datavol
        elif output_type == 'sumL2L3':
            self.predict = sum_datavolL2L3
        elif output_type == 'time':
            self.predict = time_estimate
    


class DataVolumeCostModel(CostModel):
    """

    Parameters
    ----------
    task: Task
        The tuning task
    feature_type: str, optional
        If is 'itervar', use features extracted from IterVar (loop variable).
        If is 'knob', use flatten ConfigEntity directly.
        If is 'curve', use sampled curve feature (relation feature).

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' are good.
                                'itervar' is more accurate but 'knob' is much faster.
                                There are some constraints on 'itervar', if you meet
                                problems with feature extraction when using 'itervar',
                                you can switch to 'knob'.

        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability,
                               'knob' is faster.
        For cross-device or cross-operator tuning, you can use 'curve' only.
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    upper_model: XGBoostCostModel, optional
        The upper model used in transfer learning
    """
    def __init__(self, task, feature_type, prediction_type='sumL2L3', num_threads=None,
                 upper_model=None):
        super(DataVolumeCostModel, self).__init__()

        self.task = task
        self.target = task.target
        self.space = task.config_space

        self.fea_type = feature_type
        self.num_threads = num_threads

        self.saved_features = {'scores':[]}

        self.bst = DataVolumePredictor(prediction_type)

        self.feature_extract_func = _extract_datavol_feature_index

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.pool = None
        self.base_model = None

        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        # use global variable to pass common arguments
        global _extract_space, _extract_target, _extract_task
        _extract_space = space
        _extract_target = target
        _extract_task = task
        self.pool = multiprocessing.Pool(self.num_threads)

    def _close_pool(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def fit_random(self, xs, ys, plan_size):

        x_train = self._get_feature(xs, save_features=True)

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)

        x_train = self._get_feature(xs, save_features=True)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        self._sample_size = len(x_train)

        # Get scores for testing datavol features
        if self.bst != None:
            scores = self.bst.predict(x_train)
            self.saved_features['scores'].append((xs.copy(), scores))


    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)

        return self.bst.predict(feas)

    def _get_feature(self, indexes, save_features=False):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            feas = []
            pool = self._get_pool()
            feas = pool.map(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea[0]
                if save_features:
                    if i in self.saved_features.keys():
                        self.saved_features[i].set_feature(fea[0])
                        self.saved_features[i].set_config(fea[1])
                    else:
                        self.saved_features[i] = SavedFeature(config=fea[1], feature=fea[0])

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret

    def __del__(self):
        self._close_pool()


_extract_space = None
_extract_target = None
_extract_task = None

def _extract_datavol_feature_index(index):
    config = _extract_space.get(index)
    config_features = []
    for param in config.to_json_dict()['entity']:
        if param[1] == 'sp':
            config_features += param[-1][1:]
        elif param[1] == 'an':
            for setting in param[-1]:
                config_features += [int(setting != 'none')]
    with _extract_target:
        # Get schedule and args, which initializes config to contain schedule info.
        sch, args = _extract_task.instantiate(config)
    order = [0] + (np.array(config['reorder_0'].perm) + 1).tolist()
    if config.arrays != None:
        d_foot, d_vol = estimate_dv(config['reorder_0'].perm, [50] + config.extents,
                config.array_dims, cache_sizes, config.conv_dims, config.fastest_varying, 
                arrays=config.arrays)
    else:
        d_foot, d_vol = estimate_dv(config['reorder_0'].perm, [50] + config.extents,
                config.array_dims, cache_sizes, config.conv_dims, config.fastest_varying)
    return d_vol[2][:,:,-1].sum(axis=0), config.to_json_dict()


def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_knob_feature_index(index):
    """extract knob feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        return config.get_flatten_feature()
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        x = config.get_flatten_feature()

        if res.error_no == 0:
            with inp.target:  # necessary, for calculating flops of this task
                inp.task.instantiate(config)
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_curve_feature_index(index):
    """extract sampled curve feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        with _extract_target:
            sch, args = _extract_task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return np.array(fea)
    except Exception:  # pylint: disable=broad-except
        return None

def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    try:
        inp, res = arg
        config = inp.config
        with inp.target:
            sch, args = inp.task.instantiate(config)
        fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
        x = np.concatenate((fea, list(config.get_other_option().values())))

        if res.error_no == 0:
            y = inp.task.flop / np.mean(res.costs)
        else:
            y = 0.0
        return x, y
    except Exception:  # pylint: disable=broad-except
        return None

"""
def custom_callback(stopping_rounds, metric, fevals, evals=(), log_file=None,
                    maximize=False, verbose_eval=True):
    # pylint: disable=import-outside-toplevel
    from xgboost.core import EarlyStopException
    from xgboost.callback import _fmt_metric
    from xgboost.training import aggcv

    state = {}
    metric_shortname = metric.split("-")[1]

    def init(env):
        bst = env.model

        state['maximize_score'] = maximize
        state['best_iteration'] = 0
        if maximize:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        if not state:
            init(env)

        bst = env.model
        i = env.iteration
        cvfolds = env.cvfolds

        res_dict = {}

        ##### evaluation #####
        if cvfolds is not None:
            for feval in fevals:
                tmp = aggcv([f.eval(i, feval) for f in cvfolds])
                for k, mean, std in tmp:
                    res_dict[k] = [mean, std]
        else:
            for feval in fevals:
                bst_eval = bst.eval_set(evals, i, feval)
                res = [x.split(':') for x in bst_eval.split()]
                for kv in res[1:]:
                    res_dict[kv[0]] = [float(kv[1])]

        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        ##### print eval result #####
        infos = ["XGB iter: %3d" % i]
        for item in eval_res:
            if 'null' in item[0]:
                continue
            infos.append("%s: %.6f" % (item[0], item[1]))

        if not isinstance(verbose_eval, bool) and verbose_eval and i % verbose_eval == 0:
            logger.debug("\t".join(infos))
        if log_file:
            with open(log_file, "a") as fout:
                fout.write("\t".join(infos) + '\n')

        ##### choose score and do early stopping #####
        score = None
        for item in eval_res:
            if item[0] == metric:
                score = item[1]
                break
        assert score is not None

        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d] %s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in eval_res]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback
"""


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)
    return feval

def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]
    return feval

def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N
    return feval

def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]
    return feval

def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""
    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]
    return feval

def xgb_null_score(_):
    """empty score function for xgb"""
    def feval(__, ___):
        return "null", 0
    return feval
