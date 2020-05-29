import pickle
import numpy as np
import sys

    #Works with TVM installed, otherwise load npz
file_string = sys.argv[1]
with open(file_string, 'rb') as f:
    task, data = pickle.load(f)

flat_configs = []
flat_features = []
flat_results = []

include_strings = False
keys = list(data.keys())
keys.remove('scores')

np.random.seed(0)
shuffled_indices = np.array(keys).astype(np.int32)
np.random.shuffle(shuffled_indices)
clean_indices = []

for index in shuffled_indices:
    if data[index].feature is None:
        continue
    if data[index].result is None or data[index].result.error_no > 0:
        continue

    flat_results.append(np.array(data[index].result.costs).mean())
    flat_configs.append(data[index].config.get_flatten_feature())
    flat_features.append(data[index].feature)
    clean_indices.append(index)
    #arr = []
    #for feature in data[index].feature:
    #    first_var = True
    #    if not isinstance(feature, list):
    #        arr.append(feature)
    #    else:
    #        for var in feature:
    #            if include_strings:
    #                if first_var:
    #                    arr.append(var[0])
    #                    arr.append(var[1].name)
    #                    first_var = False
    #                else:
    #                    arr.extend(var)
    #            else:
    #                if first_var:
    #                    first_var = False
    #                else:
    #                    arr.extend(var[1:])
    #flat_features.append(arr)
scores = data['scores']

np.savez(file_string[:-3]+'npz', configs=flat_configs, features=flat_features, results=flat_results, scores=scores, indices=clean_indices, task=np.array(str(task)), allow_pickle=True)
