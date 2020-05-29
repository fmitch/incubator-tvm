import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'datavol_features_1core_56_n4_1024.npz'

data = np.load(filename, allow_pickle=True)

scores = data['scores']
indices = data['indices']
results = data['results']
features = data['features']
configs = data['configs']

from IPython import embed
embed()

level = 0
for level in [0,1,2]:
    for add in [0,3,6]:
#    for other_level in [1,2]:
#        other_level = (other_level + level) % 3
#        print(level, other_level)
        plt.subplot(2,1,1)
        plt.scatter(features[:,add+level],results)#, s=features[:,6+other_level]/features[:,6+other_level].min())
#        plt.xlabel('L%i Data Volume, Size=L%i' % (level+1, other_level+1))
        plt.xlabel('L%i Data Volume' % (level+1))
        plt.ylabel('Time')
        plt.title('Full Footprint')
        #plt.show()
        plt.subplot(2,1,2)
        plt.scatter(features[:,add+9+level],results)#, s=features[:,15+other_level]/features[:,15+other_level].min())
        plt.xlabel('L%i Data Volume' % (level+1))
        plt.ylabel('Time')
        plt.title('Array Footprint')
        plt.show()



