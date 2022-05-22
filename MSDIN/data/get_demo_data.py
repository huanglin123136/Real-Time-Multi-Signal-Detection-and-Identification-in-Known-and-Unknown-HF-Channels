import numpy as np

path = 'testdata_i10_0403_0.npy'
data = np.load(path, allow_pickle=True)

new_data = data[130:140]
np.save('demo.npy', new_data)