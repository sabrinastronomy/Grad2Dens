# Core scientific stack
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import arviz as az
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew, kurtosistest, normaltest
import jax
# Local modules
import alternating, jax_battaglia_full, theory_matter_ps, jax_main

### defaults for paper plots
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})

k_0_fiducial = 0.185 * 0.676 # changing from Mpc/h to Mpc
alpha_fiducial = 0.564
b_0_fiducial = 0.593
midpoint_z_fiducial = 7
tanh_fiducial = 100

fiducial_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tanh_fiducial,
                   "avg_z": midpoint_z_fiducial}  # b_0=0.5, alpha=0.2, k_0=0.1)
static_redshift = True
ska_effects = False
nominal = True

bins = 32
side_length = 32
physical_side_length = 64
resolution = physical_side_length / side_length
dimensions = 2
area = physical_side_length**dimensions
z = 7

key = jax.random.PRNGKey(0)
dims = tuple([side_length] * dimensions)
starting_field = jax.random.normal(key, dims)

iter_num_max = 1
rest_num_max = 1

nothing_off = True
cov_matrix_data = False
# truth_field_full = jnp.load(f"21cmfast_1_Mpcpp_256_num_pixels/density_{z}.npy")
# truth_field = truth_field_full[:side_length, :side_length, 0] #, :side_length]
# truth_field_full = truth_field

def fun_corr_coeff(field_1, field_2, bins, resolution, area):
    counts, pspec_truth, bin_means = theory_matter_ps.circular_spec_normal(truth_field_full, bins, resolution, area)
    # counts, pspec_hi, bin_means = theory_matter_ps.circular_spec_normal(x_hi_truth, bins, resolution, area)
    counts, pspec_inferred, bin_means = theory_matter_ps.circular_spec_normal(DensityBlock.best_field_reshaped, bins, resolution, area)
    counts, cross_inferred, bin_means = theory_matter_ps.CROSS_circular_spec_normal(truth_field_full, DensityBlock.best_field_reshaped, bins, resolution, area)

    r_12 = cross_inferred/np.sqrt(pspec_inferred*pspec_truth)
    print(r_12)
    return jnp.sum(r_12)


# kurt_truth = kurtosis(truth_field.flatten(), fisher=True)
# skew_truth = skew(truth_field.flatten())

# print("kurtosis, skew")
# print(kurt_truth, skew_truth)
ionized_frac = []
corr_coeff = []
for z in jnp.arange(6, 8, 0.1):
    brightness_temperature_field = []

    params = alternating.ConfigParam(ska_effects=ska_effects, free_params=fiducial_params, z=z, truth_field=[],
                                     brightness_temperature_field=brightness_temperature_field, num_bins=bins,
                                     nothing_off=nothing_off, plot_direc="", side_length=side_length,
                                     physical_side_length=physical_side_length,
                                     dimensions=dimensions, iter_num_max=iter_num_max, rest_num_max=rest_num_max,
                                     noise_off=True,
                                     new_prior=False, old_prior=True, cov_matrix_data=False, run_optimizer=True)
    DensityBlock = alternating.InferDens(config_params=params, s_field=None)
    plt.close()
    plt.imshow(DensityBlock.data)
    count = jnp.sum(DensityBlock.data < 16)
    ion_frac = count / float(side_length ** dimensions)

    # counts, pspec_cross, bin_means = theory_matter_ps.CROSS_circular_spec_normal(truth_field_full, x_hi_truth, bins, resolution, area)
    truth_field_full = DensityBlock.truth_field
    ionized_frac.append(ion_frac)
    cc = fun_corr_coeff(DensityBlock.best_field_reshaped, truth_field_full, bins, resolution, area)
    corr_coeff.append(cc)


    print("-----ion frac------")
    print(corr_coeff)
    print(ionized_frac)
