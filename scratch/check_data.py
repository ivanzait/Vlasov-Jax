import numpy as np

data_fine = np.load('data/fine/step_0000.npz')
data_coarse = np.load('data/coarse/step_0000.npz')

print("Fine keys:", list(data_fine.keys()))
print("Fine f shape:", data_fine['f'].shape)
print("Coarse keys:", list(data_coarse.keys()))
print("Coarse f shape:", data_coarse['f'].shape)
