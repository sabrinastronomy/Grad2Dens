import numpy as np
import matplotlib.pyplot as plt

z = 8
side_length = 256
plot_direc_new="data_adversarial_new_prior_256_bins"
plot_direc_old="data_adversarial_old_prior_256_bins"

truth_field = np.load(plot_direc_new + f"/npy/truth_field_{z}.npy")
classic_pri_field = np.load(plot_direc_new + f"/npy/likelihood_off_best_field_8_FINAL.npy")
new_pri_field = np.load(plot_direc_old + f"/npy/likelihood_off_best_field_8_FINAL.npy")


def quick_p_spec_normal(field, side_length, nbins=256):
    """
    square before averaging (histogramming)
    """
    assert np.shape(field) == (side_length, side_length)
    fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    fft_data_squared = np.abs(fft_data) ** 2
    k_arr = np.fft.fftshift(np.fft.fftfreq(side_length)) * 2 * np.pi
    k1, k2 = np.meshgrid(k_arr, k_arr)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)

    counts, bin_edges = np.histogram(k_mag_full, nbins)
    binned_power, _ = np.histogram(k_mag_full, nbins, weights=fft_data_squared)
    kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pspec = binned_power / counts / (side_length ** 2)
    return counts, pspec, kvals

_, pspec_truth, kvals = quick_p_spec_normal(truth_field, side_length)
_, pspec_classic, kvals = quick_p_spec_normal(classic_pri_field, side_length)
_, pspec_new, kvals = quick_p_spec_normal(new_pri_field, side_length)


fig_pspec, axes_pspec = plt.subplots()
axes_pspec.loglog(kvals, pspec_truth, label=f"Truth field", c="k")
axes_pspec.loglog(kvals, pspec_classic, label=f"Classic prior", ls="--")
axes_pspec.loglog(kvals, pspec_new, label=f"New prior", ls="--")
plt.legend()
plt.savefig("last_step_prior_comp.png")