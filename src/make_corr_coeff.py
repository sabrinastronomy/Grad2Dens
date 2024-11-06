import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

num_bins = 32
dim = 3
seed = 1
correlation_coeffs = []

alphas = np.linspace(0.1, 2, 20)
b_0s = np.linspace(0.1, 2, 20)
k_0s = np.linspace(0.1, 2, 20)
avg_zs = np.linspace(6, 7, 20)

k_0_fiducial = 0.185
alpha_fiducial = 0.564
b_0_fiducial = 0.593
midpoint_z_fiducial = 7
tanh_fiducial = 1

free_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tanh_fiducial,
                   "avg_z": midpoint_z_fiducial, "redshift_run": 6.5}  # b_0=0.5, alpha=0.2, k_0=0.1)
z_values = [6.5, 7, 7.5]
for z in z_values:

    str_free_params = "_".join([f"{key}-{value}" for key, value in free_params.items()])

    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_perc_ionized_*_seed_{seed}_bins_{num_bins}"
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_diff_start_" + new_direc

    matching_dirs = glob.glob(new_direc)
    new_direc = matching_dirs[0]
    # File paths
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields
    flat_field1 = field1.flatten()
    flat_field2 = field2.flatten()

    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(flat_field1, flat_field2)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coeffs.append(correlation_coefficient)
colors = ['red' if coef < 0 else 'green' if coef > 0.7 else 'orange' for coef in correlation_coeffs]

plt.figure(figsize=(8, 5))

# Bar plot showing correlation strength
sns.barplot(x=z_values, y=correlation_coeffs, palette=colors)

# Add title and labels
plt.title('Correlation Coefficient vs Redshift Values')
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
plt.savefig("z_corr_coeff.png")
plt.close()
correlation_coeffs = []
z = 6.5
for alpha in alphas:
    free_params["alpha"] = np.round(alpha, decimals=4)
    str_free_params = "_".join([f"{key}-{value}" for key, value in free_params.items()])
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_perc_ionized_*_seed_{seed}_bins_{num_bins}"
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_diff_start_" + new_direc

    matching_dirs = glob.glob(new_direc)
    new_direc = matching_dirs[0]
    # File paths
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields
    flat_field1 = field1.flatten()
    flat_field2 = field2.flatten()

    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(flat_field1, flat_field2)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coeffs.append(correlation_coefficient)
colors = ['red' if coef < 0 else 'green' if coef > 0.7 else 'orange' for coef in correlation_coeffs]

plt.figure(figsize=(8, 5))

# Bar plot showing correlation strength
sns.barplot(x=alphas, y=correlation_coeffs, palette=colors)

# Add title and labels
plt.title('Correlation Coefficient vs Alpha Values')
plt.xlabel('Redshift (z)')
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
plt.savefig("alpha_corr_coeff.png")
plt.close()
z = 6.5
correlation_coeffs = []

for b_0 in b_0s:
    free_params["b_0"] = np.round(b_0, decimals=4)

    str_free_params = "_".join([f"{key}-{value}" for key, value in free_params.items()])
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_perc_ionized_*_seed_{seed}_bins_{num_bins}"
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_diff_start_" + new_direc

    matching_dirs = glob.glob(new_direc)
    new_direc = matching_dirs[0]
    # File paths
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields
    flat_field1 = field1.flatten()
    flat_field2 = field2.flatten()

    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(flat_field1, flat_field2)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coeffs.append(correlation_coefficient)
colors = ['red' if coef < 0 else 'green' if coef > 0.7 else 'orange' for coef in correlation_coeffs]

plt.figure(figsize=(8, 5))

# Bar plot showing correlation strength
sns.barplot(x=alphas, y=correlation_coeffs, palette=colors)

# Add title and labels
plt.title('Correlation Coefficient vs Beta Values')
plt.xlabel('b0s')
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
plt.savefig("beta_corr_coeff.png")
plt.close()
correlation_coeffs = []

for k_0 in k_0s:
    free_params["k_0"] = np.round(k_0, decimals=4)

    str_free_params = "_".join([f"{key}-{value}" for key, value in free_params.items()])
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_perc_ionized_*_seed_{seed}_bins_{num_bins}"
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_diff_start_" + new_direc

    matching_dirs = glob.glob(new_direc)
    new_direc = matching_dirs[0]
    # File paths
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields
    flat_field1 = field1.flatten()
    flat_field2 = field2.flatten()

    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(flat_field1, flat_field2)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coeffs.append(correlation_coefficient)
colors = ['red' if coef < 0 else 'green' if coef > 0.7 else 'orange' for coef in correlation_coeffs]

plt.figure(figsize=(8, 5))

# Bar plot showing correlation strength
sns.barplot(x=k_0s, y=correlation_coeffs, palette=colors)

# Add title and labels
plt.title('Correlation Coefficient vs k0 Values')
plt.xlabel('k0s')
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
plt.savefig("k0s_corr_coeff.png")
plt.close()
correlation_coeffs = []
z = 6.5
for avg_z in avg_zs:
    free_params["avg_z"] = np.round(avg_z, decimals=4)

    str_free_params = "_".join([f"{key}-{value}" for key, value in free_params.items()])
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_perc_ionized_*_seed_{seed}_bins_{num_bins}"
    new_direc = f"free_params_{str_free_params}_dimensions_{dim}_z_{z}_diff_start_" + new_direc

    matching_dirs = glob.glob(new_direc)
    new_direc = matching_dirs[0]
    # File paths
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields
    flat_field1 = field1.flatten()
    flat_field2 = field2.flatten()

    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(flat_field1, flat_field2)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coeffs.append(correlation_coefficient)
colors = ['red' if coef < 0 else 'green' if coef > 0.7 else 'orange' for coef in correlation_coeffs]

plt.figure(figsize=(8, 5))

# Bar plot showing correlation strength
sns.barplot(x=avg_zs, y=correlation_coeffs, palette=colors)

# Add title and labels
plt.title('Correlation Coefficient vs Midpoint of Reionization')
plt.xlabel('midpoint of reionization')
plt.ylabel('Correlation Coefficient')
plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
plt.savefig("mid_z_corr_coeff.png")
plt.close()