import numpy as np
import matplotlib.pyplot as plt
side_length = 16
data_cov= np.load(f"cov_matrix/data_cov_{side_length}.npy")
data_cov_inv = np.load(f"cov_matrix/data_cov_inv_{side_length}.npy")

print("Conditional Data Cov")
print(np.linalg.cond(data_cov))

print("C^-1 C ~ I?")
difference_cov = np.matmul(np.linalg.inv(data_cov), data_cov)
identity_error = np.linalg.norm(difference_cov - np.eye(difference_cov.shape[0]))
print(identity_error)
print(difference_cov)
plt.matshow(data_cov)
plt.colorbar()
plt.savefig(f"cov_matrix/data_cov_{side_length}.png")
