import matplotlib.pyplot as plt
import numpy as np

dv_data = np.load('datavol_features_1core_56_n4_2048.npz', allow_pickle=True)
iv_data = np.load('itervar_features_1core_56_n4_2048.npz', allow_pickle=True)

dv_scores = dv_data['scores']
dv_indices = dv_data['indices']
dv_results = dv_data['results']
iv_scores = iv_data['scores']
iv_indices = iv_data['indices']
iv_results = iv_data['results']
dv_corrs = []
iv_corrs = []

for i in range(len(dv_scores)):
    inds = []
    scores = []
    all_inds, all_scores = dv_scores[i]
    for ind, score in zip(all_inds, all_scores):
        where = np.where(dv_indices==ind)
        if len(where[0]) > 0:
            inds.append(where[0][0])
            scores.append(score)
    plt.scatter(dv_results[inds], scores,label='DataVol')
    dv_corrs.append(np.corrcoef(dv_results[inds], scores)[0][1])

    inds = []
    scores = []
    all_inds, all_scores = iv_scores[i]
    for ind, score in zip(all_inds, all_scores):
        where = np.where(iv_indices==ind)
        if len(where[0]) > 0:
            inds.append(where[0][0])
            scores.append(score)
    plt.scatter(iv_results[inds], scores,label='IterVar')
    iv_corrs.append(np.corrcoef(iv_results[inds], scores)[0][1])
    print('IV', np.corrcoef(iv_results[inds], scores))
    plt.xlabel('Time')
    plt.ylabel('Predicted Score')
    plt.legend()
    if i%5 == 0:
        plt.show()
    else:
        plt.close()
plt.plot(dv_corrs,label='DataVol Corr Coef')
plt.plot(iv_corrs,label='IterVar Corr Coef')
plt.xlabel('Training Iteration')
plt.show()

