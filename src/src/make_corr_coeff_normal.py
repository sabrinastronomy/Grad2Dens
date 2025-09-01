import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from theory_matter_ps import spherical_p_spec_normal
import os
import re

dim = 3
num_bins = 32
seed = 1

def normalize_filename(filename):
    # Replace the variable part in the filename with a placeholder
    # Example: "perc_ionized_0.0" -> "perc_ionized"
    return re.sub(r'perc_ionized_\d+(\.\d+)?', 'perc_ionized', filename)

def find_matching_files(wo_ska, ska):
    wo_ska_filenames = np.sort(np.asarray(os.listdir(wo_ska)))
    ska_filenames = np.sort(np.asarray(os.listdir(ska)))

    print(wo_ska_filenames[:5])
    print(ska_filenames[:5])

    normalized_files1 = [normalize_filename(f) for f in wo_ska_filenames]
    normalized_files2 = [normalize_filename(f) for f in ska_filenames]
    print(normalized_files1[0], normalized_files2[0])
    # Remove the prefix from files in dir2 before comparison
    prefix = "ska_on_"
    files2_new = [filename[len(prefix):] for filename in normalized_files2 if filename.startswith(prefix)]
    normalized_files1, files2_new = np.asarray(normalized_files1), np.asarray(files2_new)
    # Find matching files
    matching_files = np.intersect1d(normalized_files1, files2_new)
    print("matching_files", matching_files)
    new_wo_ska = []
    new_w_ska = []

    for match in matching_files:
        print("match")
        print(match)
        mask_wo_ska = normalized_files1 == match
        el = wo_ska_filenames[mask_wo_ska]
        assert(len(el) == 1)
        new_wo_ska.append(el[0])

        mask_w_ska = files2_new == match
        el = ska_filenames[mask_w_ska]
        assert(len(el) == 1)
        new_w_ska.append(el[0])

    return new_wo_ska, new_w_ska


def extract_parameters(filename):
    # Adjusted regex pattern to capture `b_0`, remove underscores before the key, and capture perc_ionized
    pattern = r'([a-zA-Z]+(?:_\d*)?)-([\d.-]+)'

    # Find all matches of the pattern in the filename
    parameters = re.findall(pattern, filename)

    # Convert the list of tuples into a dictionary
    param_dict = {key.replace('_', ''): value for key, value in parameters}

    return param_dict



# Extract parameters


# Function to calculate correlation coefficients
def compute_correlation_pspec_resid(direc, filename, z):
    new_direc = direc + filename
    # Load data
    best_field_path = f"{new_direc}/npy/best_field_{z}_FINAL.npy"
    truth_field_path = f"{new_direc}/npy/truth_field_{z}.npy"
    field1 = np.load(best_field_path)
    field2 = np.load(truth_field_path)

    # Flatten fields and compute correlation coefficient
    flat_field1 = field1.flatten() / np.std(field1)
    flat_field2 = field2.flatten() / np.std(field2)
    arr_valid = np.correlate(flat_field1, flat_field2)

    _, pspec_1, _ = spherical_p_spec_normal(field1, 32, 2, np.shape(field1)[0]**3)
    _, pspec_2, _ = spherical_p_spec_normal(field2, 32, 2, np.shape(field2)[0]**3)
    pspec_residual = np.sum(pspec_2-pspec_1)
    # print(pspec_residual)
    return arr_valid[0], pspec_residual

direc_wo_ska = "/fred/oz113/sberger/paper_1_density/Grad2Dens/src/ska_off_full_grid/"
direc_ska = "/fred/oz113/sberger/paper_1_density/Grad2Dens/src/ska_on_full_grid/"

new_wo_ska, new_w_ska = find_matching_files(direc_wo_ska, direc_ska)

# Iterate over each combination of parameters
r_arr_ska_wo = np.zeros_like(new_wo_ska)
r_arr_ska = np.zeros_like(new_w_ska)

pspec_resid_arr_ska_wo = np.zeros_like(new_wo_ska)
pspec_resid_arr_ska = np.zeros_like(new_w_ska)
k0s = []
alphas = []
b_0s = []
avg_zs = []
for i, (file_wo, file_w) in enumerate(zip(new_wo_ska, new_w_ska)):
    file_w = file_w
    params_wo = extract_parameters(file_wo)
    params_w = extract_parameters(file_w)
    # print(params_wo)
    assert(params_wo == params_w)
    k0s.append(params_wo['k0'])
    alphas.append(params_wo['alpha'])
    b_0s.append(params_wo['b0'])
    avg_zs.append(params_wo['z'])

    r_arr_ska_wo[i], pspec_resid_arr_ska_wo[i] = compute_correlation_pspec_resid(direc_wo_ska, file_wo, 6.5)
    r_arr_ska[i], pspec_resid_arr_ska[i] = compute_correlation_pspec_resid(direc_ska, file_w, 6.5)
    print(i)
    print(f"out of {len(new_wo_ska)}")

plt.semilogy(r_arr_ska_wo, label="without ska", marker="o", ls=None)
plt.semilogy(r_arr_ska, label="with ska", marker="o", ls=None)
plt.legend()
plt.ylabel("corrleation coefficient")
plt.savefig("summary_plots/corr_coeff.png")
plt.close()

plt.semilogy(pspec_resid_arr_ska_wo, label="without ska", marker="o", ls=None)
plt.semilogy(pspec_resid_arr_ska, label="with ska", marker="o", ls=None)
plt.legend()
plt.ylabel("pspec residuals")
plt.savefig("summary_plots/pspec.png")
plt.close()

# Plot and save all parameters
def plot_and_save(data, ylabel, filename):
    plt.plot(data, marker="o", ls=None)
    plt.ylabel(ylabel)
    plt.savefig(f"summary_plots/{filename}.png")
    plt.close()

# Plot each parameter
plot_and_save(alphas, "alphas", "alphas")
plot_and_save(k0s, "k0", "k0s")
plot_and_save(b_0s, "b_0", "b_0s")
plot_and_save(avg_zs, "avg_z", "avg_zs")
