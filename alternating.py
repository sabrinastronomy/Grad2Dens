"""
Infrastructure to generate MAPs of the matter density field
Created November, 2022
Written by Sabrina Berger
"""

from jax_main import SwitchMinimizer
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
import os
from matplotlib.ticker import MaxNLocator
from jax_battaglia_full import Dens2bBatt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from theory_matter_ps import circular_spec_normal, after_circular_spec_normal
from jax.experimental.host_callback import id_print  # this is a way to print in Jax when things are precompiled

### defaults for paper plots
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

class InferDens(SwitchMinimizer):
    """
    This class allows you to pass in a brightness temperature field
    measurement at a particular redshift and infer the density field.
    """
    def __init__(self, config_params, s_field):
        self.config_params = config_params
        self.s_field = s_field
        super().__init__(config_params, s_field)
        self.final_function_value_output = jnp.zeros(self.config_params.iter_num_max)
        self.mse_vals = jnp.empty(self.config_params.iter_num_max)
        self.posterior_vals = jnp.empty(self.config_params.iter_num_max)
        self.likelihood_vals = jnp.empty(self.config_params.iter_num_max)
        self.prior_vals = jnp.empty(self.config_params.iter_num_max)
        self.pixel_tracks_ionized = jnp.empty(self.config_params.iter_num_max)
        self.pixel_tracks_neutral = jnp.empty(self.config_params.iter_num_max)
        self.hessian_vals = jnp.empty(self.config_params.iter_num_max)

        self.labels = []

        if config_params.plot_direc == "":
            self.perc_ionized = round(len(self.ionized_indices) / self.config_params.side_length**self.config_params.dim, 1)
            if config_params.old_prior:
                new_direc = f"tanh_slope_{self.config_params.tanh_slope}_z_{self.config_params.z}_perc_ionized_{self.perc_ionized}_old_seed_{self.config_params.seed}_bins_{self.config_params.num_bins}"
            elif config_params.new_prior:
                new_direc = f"tanh_slope_{self.config_params.tanh_slope}_z_{self.config_params.z}_perc_ionized_{self.perc_ionized}_seed_{self.config_params.seed}_bins_{self.config_params.num_bins}"

            if self.truth_field.any() != None:
                new_direc = f"tanh_slope_{self.config_params.tanh_slope}_z_{self.config_params.z}_diff_start_" + new_direc
            try:
                os.mkdir(new_direc)
                os.mkdir(new_direc + "/plots")
                os.mkdir(new_direc + "/npy")
            except:
                print("directory already exists")
            self.plot_direc = new_direc
        else:
            self.plot_direc = self.config_params.plot_direc # this overwrites the plot_directory in the SwitchMinimizer class

        print("Saving plots here...")
        print(self.plot_direc)

        if run_optimizer:
            self.infer_density_field()
            self.make_1_1_plots()

    def check_threshold(self):
        batt_model_instance = Dens2bBatt(self.best_field_reshaped, delta_pos=1, set_z=self.config_params.z, flow=True, tanh_slope=self.config_params.tanh_slope)
        temp_model_brightness = batt_model_instance.temp_brightness
        root_mse_curr = self.root_mse(self.data, temp_model_brightness)
        return root_mse_curr

    def root_mse(self, field_mod, field_predicted):
        std_dev_field_mod = jnp.std(field_mod)
        diff = (field_mod - field_predicted) / std_dev_field_mod
        sum_pre = jnp.sum(diff**2)
        sum_2 = sum_pre ** (1/2)
        return sum_2

    def infer_density_field(self):
        rest_num = 0
        for iter_num_big in range(self.config_params.iter_num_max):
            if iter_num_big == 0:
                if self.config_params.nothing_off:
                    self.prior_off = False
                    self.likelihood_off = False
                    self.mask_ionized = False
                else:
                    # first iteration, we keep just the likelihood on
                    self.prior_off = True
                    self.likelihood_off = False
                    self.mask_ionized = False
                ####
                print("Initial run")
                if self.likelihood_off:
                    print("Likelihood off")
                else:
                    print("Likelihood ON")

                if self.prior_off:
                    print("Prior off")
                else:
                    print("Prior ON")
                self.run(likelihood_off=self.likelihood_off, prior_off=self.prior_off, mask_ionized=self.mask_ionized, use_old_field=False, iter_num_big=iter_num_big)
                ####
                self.check_field(self.s_field_original, "starting field", show=False, save=True,
                                 iteration_number=-1)
                self.check_field(self.truth_field, "truth field", show=False, save=True,
                                 iteration_number=-1)
                self.check_field(self.data, "data (brightness temperature)", show=False, save=True,
                                 iteration_number=-1)

                np.save(self.plot_direc + f"/npy/truth_field_{self.config_params.z}.npy", self.truth_field)
                np.save(self.plot_direc + f"/npy/data_field_{self.config_params.z}.npy", self.data)
            else:
                if not self.config_params.nothing_off: # if both on this is skipped
                    if rest_num == self.config_params.rest_num_max or self.prior_off:
                        self.prior_off = not self.prior_off
                        self.likelihood_off = not self.likelihood_off
                        rest_num = 0
                    if self.likelihood_off:
                        self.mask_ionized = True
                else:
                    self.mask_ionized = True

                print("Subsequent run")
                self.run(likelihood_off=self.likelihood_off, prior_off=self.prior_off, mask_ionized=self.mask_ionized, use_old_field=True, iter_num_big=iter_num_big)

            print("--------------------------------------------------------")
            if self.config_params.verbose:
                print(f"likelihood on {self.prior_off}")
                print(f"prior on {self.likelihood_off}")
                print(f"mask on {self.mask_ionized}")

            if not self.likelihood_off and not self.prior_off:
                self.plot_title = "both_on"
            elif self.likelihood_off:
                self.plot_title = "likelihood_off"
            elif self.prior_off:
                self.plot_title = "prior_off"
            else:
                print("Something is wrong as neither the likelihood or prior is on.")
                exit()

            batt_model_instance = Dens2bBatt(self.best_field_reshaped, delta_pos=1, set_z=self.config_params.z, flow=True, tanh_slope=self.config_params.tanh_slope)
            temp_model_brightness = batt_model_instance.temp_brightness

            self.check_field(temp_model_brightness, "brightness temperature", show=True, save=True,
                             iteration_number=iter_num_big)
            # self.check_field(self.best_field_reshaped-self.truth_field, "residuals", show=True, save=True,
            #                      iteration_number=iter_num_big)

            np.save(self.plot_direc + f"/npy/best_field_{self.config_params.z}_{iter_num_big}.npy", self.best_field_reshaped)
            self.labels.append(self.plot_title) # saving configuration of each iteration in list
            np.save(self.plot_direc + f"/npy/labels.npy", self.labels)

            print("iter num big")
            print(iter_num_big)
            print("likelihood off")
            print(self.likelihood_off)
            print("prior_off")
            print(self.prior_off)
            print("mask ionized")
            print(self.mask_ionized)

            rest_num += 1 # INCREASE REST NUM ONE


            # saving information only at the end of a round of likelihood/prior off
            # getting 0th ionized/neutral index and saving to array
            # ionized_ind = self.ionized_indices[0]
            # neutral_ind = self.neutral_indices[0]
            # ionized_val = self.best_field.flatten()[ionized_ind]
            # neutral_val = self.best_field.flatten()[neutral_ind]
            # self.pixel_tracks_ionized = self.pixel_tracks_ionized.at[iter_num_big].set(ionized_val)
            # self.pixel_tracks_neutral = self.pixel_tracks_neutral.at[iter_num_big].set(neutral_val)

            # getting mse vals, final function val from optimizer, likelihood vals, posterior vals
            self.mse_vals = self.mse_vals.at[iter_num_big].set(self.check_threshold())
            self.final_function_value_output = self.final_function_value_output.at[iter_num_big].set(self.final_func_val)

            if self.config_params.save_prior_likelihood_arr:
                prior, likelihood = self.calc_prior_likelihood(self.best_field_reshaped)
                self.prior_vals = self.prior_vals.at[iter_num_big].set(prior)
                self.likelihood_vals = self.likelihood_vals.at[iter_num_big].set(likelihood)
                self.posterior_vals = self.likelihood_vals.at[iter_num_big].set(prior + likelihood)

            # # save intermediate arrays of all important quantities
            np.save(self.plot_direc + f"/npy/prior_vals_{self.config_params.z}.npy", self.prior_vals)
            np.save(self.plot_direc + f"/npy/likelihood_vals_{self.config_params.z}.npy", self.likelihood_vals)
            np.save(self.plot_direc + f"/npy/posterior_vals_{self.config_params.z}.npy", self.posterior_vals)
            np.save(self.plot_direc + f"/npy/optimizer_output_vals_{self.config_params.z}.npy", self.final_function_value_output)
            np.save(self.plot_direc + f"/npy/hessian_vals_{self.config_params.z}.npy", self.hessian_vals)
            np.save(self.plot_direc + f"/npy/mse_vals_{self.config_params.z}.npy", self.mse_vals)
        self.plot_all_optimizer_vals()
        np.save(self.plot_direc + f"/npy/best_field_{self.config_params.z}_FINAL.npy",
                self.best_field_reshaped)

    def plot_mask(self):
        self.masked_field = np.copy(self.best_field)
        self.masked_field[self.neutral_indices_mask] = 0
        reshaped_masked = jnp.reshape(self.masked_field, self.size)
        plt.imshow(reshaped_masked)
        plt.title("best field masked during optimization")
        plt.savefig(self.plot_direc + f"/mask.png")

    def plot_pixels(self):
        fig_track, axes_track = plt.subplots()
        like_fig_track, like_axes_track = plt.subplots()

        axes_track.set_yscale("symlog")
        like_axes_track.set_yscale("symlog")
        axes_track.plot(self.pixel_tracks_ionized, c="red")
        axes_track.plot(self.pixel_tracks_neutral, c="green")
        ionized_ind = self.ionized_indices[0]
        neutral_ind = self.neutral_indices[0]
        axes_track.hlines(self.truth_field.flatten()[ionized_ind], 0, self.iter_num_max, label="ionized pixel", color="red")
        axes_track.hlines(self.truth_field.flatten()[neutral_ind], 0, self.iter_num_max, label="neutral pixel", color="green")
        axes_track.legend()
        fig_track.savefig(self.plot_direc + "/plots/tracking_pixels.png")

    def plot_hessian(self):
        plt.close()
        plt.plot(self.hessian_vals)
        plt.savefig("hessian.png", dpi=300)

    def plot_pspecs(self):
        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.config_params.z}.npy")
        self.num_k_modes = self.config_params.side_length
        counts, pspec_truth, kvals = circular_spec_normal(truth_field, self.config_params.num_bins, self.resolution, self.area)
        fig_pspec, axes_pspec = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig_pspec.subplots_adjust(hspace=0)

        axes_pspec[0].loglog(kvals, pspec_truth, label=r"$P_{\rm realization}$", c="purple")
        # axes_pspec[0].errorbar(kvals, pspec_truth, yerr=np.sqrt(counts), ls="--", alpha=0.3)

        axes_pspec[0].loglog(kvals, self.pspec_true(kvals), label=r"$P_{\rm theory}$", ls="--", c="black", zorder=100)
        axes_pspec[0].set_xticklabels([])  # Remove x-tic labels for the first frame
        axes_pspec[0].set_yticklabels([])  # Remove y-tic labels for the first frame

        label = labels[0].replace("_", " ")
        best_field = self.best_field_reshaped
        counts, pspec_normal, kvals = circular_spec_normal(best_field, self.config_params.num_bins, self.resolution, self.area)
        counts, pspec_pre, kvals = after_circular_spec_normal(best_field, self.config_params.num_bins, self.resolution, self.area)
        poisson_rms = np.sqrt(np.abs(pspec_pre - pspec_normal)) / np.sqrt(counts)
        axes_pspec[0].loglog(kvals, pspec_normal, label=f"Best Field", alpha=0.5, color="blue")
        axes_pspec[0].errorbar(kvals, pspec_normal, yerr=poisson_rms, alpha=0.3, color="blue")


        axes_pspec[1].plot(kvals, pspec_normal/pspec_truth, c="k", ls="--")
        # axes_pspec[1].set_title("log residuals between model pspec and theory pspec")
        axes_pspec[0].set_xscale("log")
        axes_pspec[0].set_yscale("log")
        axes_pspec[0].legend()
        axes_pspec[0].set_ylabel(fr"$P_{{\rm mm}}$ [$\rm Mpc^{{2}}$]")
        # axes_pspec[1].set_ylabel(r"$P_{\rm realization}/P_{\rm theory}$")
        axes_pspec[1].set_ylabel(r"$P_{\rm best}/P_{\rm mm}$")
        axes_pspec[1].set_xlabel(r"k [$\rm Mpc^{-1}$]")
        fig_pspec.savefig(self.plot_direc + "/plots/pspec.png", dpi=300)
        plt.close(fig_pspec)

    def plot_panel(self, tick_font_size=7, normalize=False, log=False):

        figsize = (8, 12)
        rows = (self.config_params.iter_num_max // 2) + 1
        cols = 2

        if normalize:
            truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.config_params.z}.npy")
        else:
            self.truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.config_params.z}.npy")
            self.data = np.load(self.plot_direc + f"/npy/data_field_{self.config_params.z}.npy")

        data_fig, data_axes = plt.subplots(rows, cols, figsize=figsize)
        residuals_fig, residuals_axes = plt.subplots(rows, cols, figsize=figsize)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        k = 0
        for i in range(rows):
            for j in range(cols):
                print("plotting", (i,j))
                self.plot_title = labels[k]
                if i == 0 and j == 0:
                    if normalize:
                        hist, bin_edges = np.histogram(truth_field, bins=100)
                        axes[i][j].plot(bin_edges[1:], hist)
                        axes[i][j].set_xscale('symlog')
                        axes[i][j].set_xlabel("truth field density")
                        axes[i][j].set_xlim(-10**1, 10**2)
                        continue
                    else:
                        if log:
                            im = axes[i][j].imshow(self.truth_field, norm=matplotlib.colors.SymLogNorm(linthresh=0.01)) #, vmin=-1, vmax=np.max(self.truth_field)))
                        else:
                            im = axes[i][j].imshow(self.truth_field, vmin=-1, vmax=1)
                        divider = make_axes_locatable(axes[i][j])
                        cax = divider.append_axes('bottom', size='5%', pad=0.05)
                        cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                        cbar.ax.tick_params(labelsize=tick_font_size)
                        t = axes[i][j].text(20, self.config_params.side_length - 10, f"Truth Field", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        axes[i][j].set_xticks([])
                        axes[i][j].set_aspect('equal')

                        if log:
                            im = data_axes[i][j].imshow(self.data, norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1, vmax=jnp.max(self.data)), cmap="oranges")
                        else:
                            im = data_axes[i][j].imshow(self.data, cmap="oranges")
                        divider = make_axes_locatable(data_axes[i][j])
                        cax = divider.append_axes('bottom', size='5%', pad=0.05)
                        cbar = data_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                        cbar.ax.tick_params(labelsize=tick_font_size)
                        t = data_axes[i][j].text(20, self.config_params.side_length - 10, f"Truth Data", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        data_axes[i][j].set_xticks([])
                        data_axes[i][j].set_aspect('equal')

                        if log:
                            im = residuals_axes[i][j].imshow(np.zeros_like(self.truth_field), norm=matplotlib.colors.SymLogNorm(linthresh=0.01))
                        else:
                            im = residuals_axes[i][j].imshow(np.zeros_like(self.truth_field))

                        divider = make_axes_locatable(residuals_axes[i][j])
                        cax = divider.append_axes('bottom', size='5%', pad=0.05)
                        cbar = residuals_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                        cbar.ax.tick_params(labelsize=tick_font_size)
                        t = residuals_axes[i][j].text(20, self.config_params.side_length - 10, f"Zero Residuals", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        residuals_axes[i][j].set_xticks([])
                        residuals_axes[i][j].set_aspect('equal')

                        continue

                best_field = np.load(self.plot_direc + f"/npy/best_field_{self.config_params.z}_{k}.npy")
                if normalize:
                    hist, bin_edges = np.histogram(best_field, bins=1000)
                    axes[i][j].plot(bin_edges[1:], hist)
                    axes[i][j].set_xscale('symlog')
                    axes[i][j].set_xlabel("density")
                    axes[i][j].set_xlim(-10**1, 10 **2)
                else:
                    if log:
                        im = axes[i][j].imshow(best_field,norm=matplotlib.colors.SymLogNorm(linthresh=0.01)) #, vmin=-1, vmax=jnp.max(self.truth_field)))
                    else:
                        im = axes[i][j].imshow(best_field, vmin=-1, vmax=1)
                    divider = make_axes_locatable(axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = axes[i][j].text(20, self.config_params.side_length - 10, f"Iteration #{k} \nwith " + self.plot_title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    axes[i][j].set_xticks([])
                    axes[i][j].set_aspect('equal')

                    if log:
                        im = residuals_axes[i][j].imshow((best_field - self.truth_field)/self.truth_field,
                                                     norm=matplotlib.colors.SymLogNorm(linthresh=0.01))
                    else:
                        im = residuals_axes[i][j].imshow(((best_field - self.truth_field)/self.truth_field)**2)
                    divider = make_axes_locatable(residuals_axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = residuals_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = residuals_axes[i][j].text(20, self.config_params.side_length - 10, f"(Current-Truth) Iteration #{k} \nwith " + self.plot_title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    residuals_axes[i][j].set_xticks([])
                    residuals_axes[i][j].set_aspect('equal')

                    batt_model_instance = Dens2bBatt(best_field, delta_pos=1, set_z=self.config_params.z, flow=True, tanh_slope=self.config_params.tanh_slope)
                    data = batt_model_instance.temp_brightness
                    if log:
                        im = data_axes[i][j].imshow(data, norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1))
                    else:
                        im = data_axes[i][j].imshow(data)
                    divider = make_axes_locatable(data_axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = data_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = data_axes[i][j].text(20, self.config_params.side_length - 10, f"Iteration #{k} \nwith " + self.plot_title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    data_axes[i][j].set_xticks([])
                    data_axes[i][j].set_aspect('equal')

                k += 1
                print("self.iter_num_max")
                print(self.config_params.iter_num_max)
                print("k")
                print(k)
                if k == self.config_params.iter_num_max:
                    break
            if normalize:
                fig.savefig(self.plot_direc + "/plots/normalize_field_panel_iterations.png", dpi=300)
            else:
                fig.savefig(self.plot_direc + "/plots/field_panel_iterations.png", dpi=300)
                data_fig.savefig(self.plot_direc + "/plots/data_panel_iterations.png", dpi=300)
                residuals_fig.savefig(self.plot_direc + "/plots/residuals_panel_iterations.png", dpi=300)

            plt.close(data_fig)
            plt.close(fig)
            plt.close(residuals_fig)

    def calc_prior_likelihood(self, field):
        field = jnp.reshape(field, self.size)
        assert np.shape(field) == (self.side_length, self.side_length)
        # note param_init was passed in and is a constant
        discrepancy = self.data - self.bias_field(field)
        likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
        #### prior
        if self.new_prior:
            counts, power_curr, bin_means = circular_spec_normal(field, self.num_bins, self.resolution, self.area)
            mask_high_k = bin_means < 5
            x = (self.pspec_box - power_curr).flatten()
            prior = np.sum(x[mask_high_k]**2)
        elif self.original_prior:
            # FT and get only the independent modes
            fourier_box = self.fft_jax(field)
            fourier_nums_real, fourier_nums_imag = self.independent_only_jax(fourier_box)
            real_prior = jnp.dot(fourier_nums_real ** 2, (2 / self.pspec_indep_nums_re))  # Half variance for real
            imag_prior = jnp.dot(fourier_nums_imag ** 2, (2 / self.pspec_indep_nums_im))  # Half variance for imag
            prior = real_prior + imag_prior
        return prior, likelihood

    def plot_all_optimizer_vals(self):
        """
        This function plots likelihood, prior, posterior, returned optimizer function value, and mse values loaded in,
        such that we can plot after the fact as well
        """
        ### plotting mse vals
        mse_vals = np.load(self.plot_direc + f"/npy/mse_vals_{self.config_params.z}.npy")
        prior_arr = np.load(self.plot_direc + f"/npy/prior_vals_{self.config_params.z}.npy")
        like_arr = np.load(self.plot_direc + f"/npy/likelihood_vals_{self.config_params.z}.npy")
        posterior_arr = np.load(self.plot_direc + f"/npy/posterior_vals_{self.config_params.z}.npy")
        final_function_value_output = np.load(self.plot_direc + f"/npy/optimizer_output_vals_{self.config_params.z}.npy")
        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        fig_mse, ax_mse = plt.subplots()
        ax_mse.semilogy(mse_vals, label="Mean square error", ls=None)
        ax_mse.semilogy(prior_arr, label="Prior", ls="--")
        ax_mse.semilogy(like_arr, label="Likelihood", ls="--")
        ax_mse.semilogy(posterior_arr, label="Posterior")
        ax_mse.semilogy(final_function_value_output, label="Output from minimizer", ls="--")

        for iter_num_big in range(self.config_params.iter_num_max):
            if labels[iter_num_big] == "prior_off" and iter_num_big == 0:
                plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="*", c="black", label="likelihood only, mask off")
            elif labels[iter_num_big] == "likelihood_off" and iter_num_big == 1:
                plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="o", c="black", label="prior only, mask on")
            elif labels[iter_num_big] == "prior_off":
                plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="*", c="black")
            elif labels[iter_num_big] == "likelihood_off":
                plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="o", c="black")
        ax_mse.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_mse.set_xlabel("Iteration")
        ax_mse.legend()
        fig_mse.savefig(self.plot_direc + f"/plots/prior_likelihood_mse_post_{self.config_params.z}.png", dpi=300)

    def plot_2_mse(self, mse_val_1="data_adversarial_new_prior", mse_val_2="data_adversarial_classic_prior"):
        ### plotting mse vals
        mse_vals_1 = np.load(mse_val_1 + f"/npy/mse_vals_{self.config_params.z}.npy")
        mse_vals_2 = np.load(mse_val_2 + f"/npy/mse_vals_{self.config_params.z}.npy")

        fig_mse, ax_mse = plt.subplots()
        length_of_runs = len(mse_vals_1[mse_vals_1 > 0]) - 1

        ax_mse.semilogy(mse_vals_1[:length_of_runs], label="MSE new prior", ls=None)
        ax_mse.semilogy(mse_vals_2[:length_of_runs], label="MSE old prior", ls=None)

        # ax_mse.set_xlim(None, 350)
        low_lim_ax, upper_lim_ax = ax_mse.get_ylim()[0], ax_mse.get_ylim()[1]

        ax_mse.vlines(3, low_lim_ax, upper_lim_ax, label="Iteration #3", color="blue")
        ax_mse.vlines(6, low_lim_ax, upper_lim_ax, label="Iteration #6", color="orange")
        ax_mse.vlines(9, low_lim_ax, upper_lim_ax, label="Iteration #9", color="green")
        ax_mse.vlines(12, low_lim_ax, upper_lim_ax, label="Iteration #12", color="red")

        # ax_mse.set_ylim(1, 12000)
        # print(mse_vals)
        # print("length of runs", length_of_runs)
        ax_mse.set_xlim(None, length_of_runs)
        ax_mse.set_xlabel("iteration")
        ax_mse.legend()
        fig_mse.savefig(f"compare_mse_post_{self.config_params.z}.png")

    def plot_3_panel(self):
        """This method gives a nice three panel plot showing the data, predicted field, and truth field"""
        from scipy.ndimage import gaussian_filter
        from matplotlib import colors

        print("MINIMUM VALUE OF DATA FIELD")
        print(jnp.min(self.data))
        plt.close()
        plt.hist(self.data, bins=100, density=True, histtype="stepfilled")
        plt.title("21cm values")
        plt.savefig(f"{self.plot_direc}/plots/data_hist.png")
        # Normalize the color range for Inferred density and Truth
        # norm_shared = colors.Normalize(vmin=min(np.min(self.best_field_reshaped), np.min(self.truth_field)),
        #                                vmax=max(np.max(self.best_field_reshaped), np.max(self.truth_field)))

        residual = self.truth_field - self.best_field_reshaped

        norm_shared = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min(np.min(self.best_field_reshaped), np.min(self.truth_field)),
                                       vmax=max(np.max(self.best_field_reshaped), np.max(self.truth_field)))

        norm_shared_residual = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=np.min(residual),
                                       vmax=np.max(residual))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

        # Plotting Observed data
        im1 = ax1.imshow(self.data, cmap="plasma")
        ax1.set_title("Observed data")
        ax1.set_xlabel("Pixel #")

        # Plotting Inferred density
        # smoothed_field = gaussian_filter(self.best_field_reshaped, sigma=1)
        im2 = ax2.imshow(self.best_field_reshaped, norm=norm_shared)
        ax2.set_title("Inferred density")
        ax2.set_xlabel("Pixel #")

        # Plotting Truth
        im3 = ax3.imshow(self.truth_field, norm=norm_shared)
        ax3.set_title("Truth")
        ax3.set_xlabel("Pixel #")

        # Plotting Residual (truth-best)
        im4 = ax4.imshow(residual, norm=norm_shared_residual)
        ax4.set_title("Residual (truth-best)")
        ax4.set_xlabel("Pixel #")


        # Create individual colorbars
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
        cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)

        fig.tight_layout()

        ## getting plot title
        if self.config_params.noise_off:
            fig.savefig(f"{self.plot_direc}/plots/3_panel_z_{self.config_params.z}_no_noise_.png", dpi=300)
        else:
            fig.savefig(f"{self.plot_direc}/plots/3_panel_z_{self.config_params.z}_w_noise.png", dpi=300)
        print("Saving three panel plot...")

        plt.close()



    def check_field(self, field, title, normalize=False, save=True, show=False, iteration_number=-1):
        # plotting
        fig, axes = plt.subplots()
        if normalize:
            field = (field - jnp.min(field)) / (jnp.max(field) - jnp.min(field))
        if field.ndim < 2:
            axes.plot(field)
        elif field.ndim == 2 and ("residual" in title or "brightness" in title):
            im = axes.imshow(field)
            cbar = fig.colorbar(im, ax=axes)
            if "brightness" in title:
                fig.suptitle("Mock 21cm Brightness Temperature Field")
                axes.set_xlabel("Pixel #")
                cbar.set_label(r"$T_{b}~\rm [mk]$")
                # Set the background color for the axes and the figure
                axes.set_facecolor('black')
                fig.patch.set_facecolor('black')

                # Set the color of the axis labels and ticks
                axes.xaxis.label.set_color('white')
                axes.yaxis.label.set_color('white')
                axes.tick_params(axis='x', colors='white')
                axes.tick_params(axis='y', colors='white')

                # Set the color of the colorbar labels and ticks
                cbar.ax.yaxis.label.set_color('white')
                cbar.ax.tick_params(axis='y', colors='white')
            if "residual" in title:
                fig.suptitle("residual")
        else:
            # density field
            im = axes.imshow(field, norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1))
            fig.colorbar(im, ax=axes)
        if save:
            if iteration_number >= 0:
                fig.savefig(f"{self.plot_direc}/plots/" + f"iter_num_{iteration_number}_" + f"{self.config_params.z}" + "_battaglia.png")
            else:
                fig.savefig(f"{self.plot_direc}/plots/" + f"{title}_{self.config_params.z}" + "_battaglia.png")

        if show:
            fig.show()
        plt.close(fig)


    def make_1_1_plots(self):
        # ionized_indices = self.data == 0
        plt.close()
        # plt.scatter(self.truth_field.flatten(), self.best_field.format(np.round(self.perc_ionized,4)), alpha=0.1, s=10)

        # batt_model_instance = Dens2bBatt(self.truth_field, delta_pos=1, set_z=self.config_params.z, flow=True, tanh_slope=self.config_params.tanh_slope)
        #
        # x_HII = 1- batt_model_instance.X_HI
        plt.scatter(self.truth_field.flatten(), self.best_field.flatten(), c=self.data, s=10)
        cbar = plt.colorbar()
        cbar.set_label(r"$T_{b}~\rm [mk]$")

        plt.plot(self.truth_field.flatten(), self.truth_field.flatten(), label="Truth Field", color="black")
        plt.xlabel("Truth")
        plt.ylabel("Best Field")

        plt.legend()
        plt.savefig(f"{self.plot_direc}/plots/reconstructed_vs_truth.png")

        # plt.close()
        # plt.scatter(self.truth_field.flatten()[self.ionized_indices_mask], self.best_field[self.ionized_indices], label="Ionization Fraction = {}".format(np.round(self.perc_ionized,4)), s=10)
        # plt.plot(self.truth_field.flatten()[self.ionized_indices], self.truth_field.flatten()[self.ionized_indices], label="truth", color="black")
        # plt.legend()
        # plt.xlabel("Truth")
        # plt.ylabel("Best Field")
        # plt.savefig(f"{self.plot_direc}/plots/just_ionized_reconstructed_vs_truth.png")


    def make_ionisation_level_residual_plot(self):
        plt.close()
        residual = self.truth_field.flatten() - self.best_field.flatten()
        batt_model_instance = Dens2bBatt(self.truth_field, delta_pos=1, set_z=self.config_params.z, flow=True, tanh_slope=self.config_params.tanh_slope)

        x_HII = 1- batt_model_instance.X_HI
        plt.scatter(x_HII, residual)
        plt.ylabel("residual")
        plt.xlabel("X_HII")
        plt.savefig(f"{self.plot_direc}/plots/residual_adelie.png")


class ConfigParam:
    def __init__(self, z, truth_field, brightness_temperature_field, num_bins, nothing_off, plot_direc, side_length, physical_side_length,
                 dimensions=2, iter_num_max=10, rest_num_max=3, noise_off=False,
                 save_posterior_values=False, run_optimizer=False, mse_plot_on=False,
                 weighted_prior=None, new_prior=False, old_prior=False, verbose=False,
                 debug=False, use_truth_mm=False, save_prior_likelihood_arr=False, seed=1010,
                 create_instance=False, tanh_slope=2):
        """
        :param z - the redshift you would like to create your density field at
        :param data (Default: None) - data that you're fitting your field to and will be used in your chi-squared.
                                If None, the data will be generated from the field meaning you know your truth field
                                beforehand.
        :param s_field (Default: None) - initial field to start density field optimization at.
                                If None, a Gaussian normal field will be used with sigma = 0.2 *
                                the standard deviation of the data.
        :param fixed_field (Default: None) - only used if include_param = True to keep the field constant while chi squared is
                                minimized and the bias is found. If None, this is not used.
        :param noise_off (Default: True) - adds noise to the data if False
        :param side_length (Default: 256) - sets the side length of the field
        :param dim (Default: 2) - sets dimensions of the field
        :param indep_prior (Default: False) - use Adrian's prior that only grabs the independent parts of the fourier transform and power spectrum to generate a prior
        :param debug (Default: False) - turns off Jit, saves all intermediate likelihood and prior values
        :param verbose (Default: False) - whether or not to print a bunch of stuff
        :param plot_direc (Default: 2D_plots) - where to save plots
        :param autorun (Default: True) - run an immediate gradient descent
        """
        self.z = z
        self.truth_field = truth_field
        self.data = brightness_temperature_field
        self.num_bins = num_bins
        self.nothing_off = nothing_off
        self.plot_direc = plot_direc
        self.side_length = side_length
        self.physical_side_length = physical_side_length
        self.dim = dimensions
        self.iter_num_max = iter_num_max
        self.rest_num_max = rest_num_max
        self.noise_off = noise_off
        self.save_posterior_values = save_posterior_values
        self.run_optimizer = run_optimizer
        self.mse_plot_on = mse_plot_on
        self.weighted_prior = weighted_prior
        self.new_prior = new_prior
        self.old_prior = old_prior
        self.verbose = verbose
        self.debug = debug
        self.use_truth_mm = use_truth_mm
        self.save_prior_likelihood_arr = save_prior_likelihood_arr
        self.seed = seed
        self.create_instance = create_instance
        self.tanh_slope = tanh_slope
        assert(self.new_prior != self.old_prior)


if __name__ == "__main__":
    print("starting")
    run_optimizer = True
    slopes = [2]
    for slope in slopes:
        for z in [6.5, 7, 8]:
            params = ConfigParam(z=z, truth_field=None,  brightness_temperature_field=None, num_bins=60,
                                 nothing_off=False, plot_direc="", side_length=128, physical_side_length=64,
                                 dimensions=2, iter_num_max=1, rest_num_max=3, noise_off=True,
                                 save_posterior_values=False, run_optimizer=True, mse_plot_on=False,
                                 weighted_prior=None, new_prior=True, old_prior=False, verbose=False,
                                 debug=False, use_truth_mm=True, save_prior_likelihood_arr=False, seed=1,
                                 create_instance=False, tanh_slope=slope)
            samp = InferDens(params, s_field=None) # setting s-field to none for first iteration
            # samp.plot_panel(normalize=False, log=True)
            samp.make_ionisation_level_residual_plot()
            samp.plot_3_panel()
            samp.plot_mask()
            samp.plot_pspecs()
