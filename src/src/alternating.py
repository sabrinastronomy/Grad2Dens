"""
Infrastructure to generate MAPs of the matter density field
Created November, 2022
Written by Sabrina Berger
"""
import time
try:
    from .jax_main import SwitchMinimizer
except:
    from jax_main import SwitchMinimizer

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
import os
from matplotlib.ticker import MaxNLocator
try:
    from .jax_battaglia_full import Dens2bBatt
    from .theory_matter_ps import spherical_p_spec_normal, after_spherical_p_spec_normal, circular_spec_normal, after_circular_spec_normal
except:
    from jax_battaglia_full import Dens2bBatt
    from theory_matter_ps import spherical_p_spec_normal, after_spherical_p_spec_normal, circular_spec_normal, after_circular_spec_normal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from scipy.ndimage import gaussian_filter
from matplotlib import colors
from scipy.stats import kurtosis, skew, kurtosistest, normaltest

# # Set up the argument parser
# parser = argparse.ArgumentParser(description="Run simulation with or without SKA effects")
#
# # Add an argument for ska_effects
# parser.add_argument(
#     '--ska_effects',
#     action='store_true',
#     help="Enable SKA effects if this flag is present"
# )
#
# parser.add_argument(
#     '--num_combinations',
#     type=int,
#     default=1,
#     help="Number of combinations we stopped at (default is 1)"
# )

# # Parse the arguments
# args = parser.parse_args()
#
# # Use the argument
# ska_effects = args.ska_effects
# num_combinations_start = args.num_combinations


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
        self.final_function_value_output = jnp.zeros(self.config_params.iter_num_max)
        self.mse_vals = jnp.empty(self.config_params.iter_num_max)
        self.posterior_vals = jnp.empty(self.config_params.iter_num_max)
        self.likelihood_vals = jnp.empty(self.config_params.iter_num_max)
        self.prior_vals = jnp.empty(self.config_params.iter_num_max)
        self.pixel_tracks_ionized = jnp.empty(self.config_params.iter_num_max)
        self.pixel_tracks_neutral = jnp.empty(self.config_params.iter_num_max)
        self.hessian_vals = jnp.empty(self.config_params.iter_num_max)
        super().__init__(config_params, s_field)
        self.labels = []
        self.iter_failed_line_search_arr = []
        if config_params.plot_direc == "":
            self.perc_ionized = len(self.ionized_indices_flattened) / self.config_params.side_length**self.config_params.dim
            str_free_params = "_".join([f"{key}-{value}" for key, value in self.config_params.free_params.items()])

            if config_params.old_prior:
                new_direc = f"free_params_{str_free_params}_dimensions_{self.config_params.dim}_z_{self.config_params.z}_perc_ionized_{self.perc_ionized}_old_seed_{self.config_params.seed}_bins_{self.config_params.num_bins}"
            elif config_params.new_prior:
                new_direc = f"free_params_{str_free_params}_dimensions_{self.config_params.dim}_z_{self.config_params.z}_perc_ionized_{self.perc_ionized}_seed_{self.config_params.seed}_bins_{self.config_params.num_bins}"


            if self.truth_field.any() != None and not self.config_params.ska_effects:
                new_direc = f"ska_off_full_grid/diff_start_" + new_direc
            elif self.truth_field.any() != None and self.config_params.ska_effects:
                new_direc = f"ska_on_diff_start_" + new_direc
            try:
                os.mkdir(new_direc)
                os.mkdir(new_direc + "/plots")
                os.mkdir(new_direc + "/npy")
            except Exception as e:
                print(e)
                print("directory already exists")
            self.plot_direc = new_direc
            self.config_params.plot_direc = new_direc
        else:
            self.plot_direc = self.config_params.plot_direc # this overwrites the plot_directory in the SwitchMinimizer class

        print("Saving plots here...")
        print(self.plot_direc)
        print("Saved config parameters.")
        self.config_params.save_to_file(directory=self.plot_direc)
        if self.config_params.run_optimizer:
            self.infer_density_field()
            self.make_1_1_plots()

    def check_threshold(self):
        start_time = time.time()
        batt_model_instance = Dens2bBatt(self.best_field_reshaped, resolution=self.resolution, set_z=self.config_params.z, physical_side_length=self.config_params.physical_side_length, flow=True, free_params=self.config_params.free_params, apply_ska=self.config_params.ska_effects)
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
                if not self.config_params.nothing_off and self.config_params.know_neutral_pixels:
                    self.mask_ionized = True
                    self.prior_off = True
                    self.likelihood_off = False
                if self.likelihood_off:
                    print("Likelihood off")
                else:
                    print("Likelihood ON")

                if self.prior_off:
                    print("Prior off")
                else:
                    print("Prior ON")
                self.run(likelihood_off=self.likelihood_off, prior_off=self.prior_off, mask_ionized=self.mask_ionized, use_old_field=False, iter_num_big=iter_num_big)

                # self.check_field(self.s_field_original, "starting field", show=False, save=True,
                #                  iteration_number=-1)
                self.check_field(self.truth_field, "truth field", show=False, save=True,
                                 iteration_number=-1)
                self.check_field(self.data, "data", show=False, save=True,
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
                    # Never masking when both are on!!
                    self.mask_ionized = False

                if not self.config_params.nothing_off and self.config_params.know_neutral_pixels:
                    self.mask_ionized = True
                    self.prior_off = False
                    self.likelihood_off = True
                print("Subsequent run")
                if self.likelihood_off:
                    print("Likelihood off")
                else:
                    print("Likelihood ON")

                if self.prior_off:
                    print("Prior off")
                else:
                    print("Prior ON")
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

            np.save(self.plot_direc + f"/npy/best_field_{self.config_params.z}_{iter_num_big}.npy", self.best_field_reshaped)
            self.labels.append(self.plot_title) # saving configuration of each iteration in list
            np.save(self.plot_direc + f"/npy/labels.npy", self.labels)
            np.save(self.plot_direc + f"/npy/iter_failed_line_search_arr.npy", self.iter_failed_line_search_arr)

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
            # self.final_function_value_output = self.final_function_value_output.at[iter_num_big].set(self.final_func_val)

            if self.config_params.save_prior_likelihood_arr:
                posterior, likelihood, prior = self.chi_sq_jax(self.best_field, debug=True)
                self.prior_vals = self.prior_vals.at[iter_num_big].set(prior)
                self.likelihood_vals = self.likelihood_vals.at[iter_num_big].set(likelihood)
                self.posterior_vals = self.posterior_vals.at[iter_num_big].set(posterior)

            # # save intermediate arrays of all important quantities
            np.save(self.plot_direc + f"/npy/prior_vals_{self.config_params.z}.npy", self.prior_vals)
            np.save(self.plot_direc + f"/npy/likelihood_vals_{self.config_params.z}.npy", self.likelihood_vals)
            np.save(self.plot_direc + f"/npy/posterior_vals_{self.config_params.z}.npy", self.posterior_vals)
            np.save(self.plot_direc + f"/npy/optimizer_output_vals_{self.config_params.z}.npy", self.final_function_value_output)
            np.save(self.plot_direc + f"/npy/hessian_vals_{self.config_params.z}.npy", self.hessian_vals)
            np.save(self.plot_direc + f"/npy/mse_vals_{self.config_params.z}.npy", self.mse_vals)
        np.save(self.plot_direc + f"/npy/best_field_{self.config_params.z}_FINAL.npy",
                self.best_field_reshaped)

        # Flatten fields
        flat_field1 = self.truth_field.flatten()
        flat_field2 = self.best_field.flatten()

        # Compute correlation coefficient
        correlation_matrix = np.corrcoef(flat_field1, flat_field2)
        correlation_coefficient = np.round(correlation_matrix[0, 1], 2)

        plt.close("all")
        plt.title(f"PDF of Density Fields, r={correlation_coefficient}")
        plt.hist(self.truth_field.flatten(), density=True, bins=100, label="Truth", alpha=0.4)
        plt.hist(self.best_field.flatten(), density=True, bins=100, label="Best", alpha=0.4)
        plt.legend()
        plt.savefig(self.plot_direc + f"/density_hist.png")
        plt.close("all")
        plt.hist(self.data.flatten(), density=True, bins=100, alpha=0.4)
        plt.title(f"PDF of Data, r={correlation_coefficient}")
        plt.savefig(self.plot_direc + f"/data_hist.png")
        plt.close("all")

    def plot_mask(self, slice_idx=10):
        plt.close()
        self.masked_field = np.copy(self.best_field_reshaped)
        self.masked_field[self.neutral_indices_mask_SHAPED] = 0
        if self.config_params.dim == 3:
            reshaped_masked = self.masked_field[slice_idx, :, :]
        elif self.config_params.dim == 2:
            reshaped_masked = self.masked_field
        else:
            print("Dimension not supported")
            exit()
        plt.imshow(reshaped_masked)
        plt.colorbar()
        plt.title("Inferred Ionized Pixel Values (all others masked)")
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
        plt.close("all")
        fig, (axes_pspec, axes_residual) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.config_params.z}.npy")
        self.num_k_modes = self.config_params.side_length

        if self.config_params.dim == 3:
            truth_field_full = np.copy(truth_field)
        if self.config_params.dim == 2:
            counts, pspec_truth, kvals = circular_spec_normal(truth_field, self.config_params.num_bins, self.resolution, self.area)
        elif self.config_params.dim == 3:
            counts, pspec_truth, kvals = spherical_p_spec_normal(self.truth_field, self.config_params.num_bins, self.resolution, self.volume)

        self.best_field = jnp.reshape(self.best_field, (self.size))
        axes_pspec.loglog(kvals, pspec_truth, label=f"Truth Field", color="red", alpha=0.5)
        axes_pspec.loglog(kvals, self.truth_final_pspec, label=f"Truth CMB Derived", alpha=0.5, color="purple", ls="--")

        if self.config_params.dim == 2:
            counts, pspec_normal, kvals = circular_spec_normal(self.best_field, self.config_params.num_bins, self.resolution, self.area)
            counts, pspec_pre, kvals = after_circular_spec_normal(self.best_field, self.config_params.num_bins, self.resolution, self.area)
        elif self.config_params.dim == 3:
            counts, pspec_normal, kvals = spherical_p_spec_normal(self.best_field, self.config_params.num_bins, self.resolution, self.volume)
            counts, pspec_pre, kvals = after_spherical_p_spec_normal(self.best_field, self.config_params.num_bins, self.resolution, self.volume)

        # Calculate the upper and lower bounds for the shaded region
        poisson_rms = np.sqrt(np.abs(pspec_pre - pspec_normal)) / np.sqrt(counts)
        upper_bound = pspec_normal + poisson_rms
        lower_bound = pspec_normal - poisson_rms

        # Plot the nominal values and the shaded region
        axes_pspec.fill_between(kvals, lower_bound, upper_bound, alpha=0.3, color="red")
        axes_pspec.loglog(kvals, pspec_normal, label=f"Best Field", alpha=0.5, color="blue")
        difference = np.sqrt((pspec_normal - pspec_truth)**2)

        # Residual plot
        # frame2 = fig1.add_axes((.1, .1, .8, .2))
        axes_residual.loglog(kvals, difference, color="black", alpha=1)
        axes_residual.set_xlabel(r"$\rm \mathbf{k}~[Mpc^{-3}]$")
        axes_pspec.set_ylabel(r"$\rm P_{mm}~[Mpc^3]$")
        axes_residual.set_ylabel(r"$\rm P_{best} - P_{truth}~[Mpc^3]$")
        # axes_pspec.set_xticklabels([])  # Remove x-tic labels for the first frame
        # low_lim_ax, upper_lim_ax = axes_pspec.get_ylim()[0], axes_pspec.get_ylim()[1]
        #
        # axes_pspec.vlines(x=2*np.pi / std_dev_beam_con, ymin=0.1, ymax=1, color="black")
        # axes_residual.vlines(x=2*np.pi / std_dev_beam_con, ymin=0.1, ymax=1, color="black")
        # axes_pspec.set_xlim((None, 5))
        # axes_residual.set_xlim((None, 5))

        ymin_pspec, ymax_pspec = axes_pspec.get_ylim()
        ymin_residual, ymax_residual = axes_residual.get_ylim()
        std_dev_beam_con = 0.6006098751085033 * 4 * np.sqrt(self.config_params.dim) # sorry i know this is bad lol
        k_vertical = 2 * np.pi / std_dev_beam_con
        if self.config_params.dim == 1:
            k_vertical = np.abs(k_vertical)
        elif self.config_params.dim == 2:
            k_vertical = np.sqrt(k_vertical ** 2 + k_vertical ** 2)
        elif self.config_params.dim == 3:
            k_vertical = np.sqrt(k_vertical ** 2 + k_vertical ** 2 + k_vertical ** 2)
        # axes_pspec.vlines(x=k_vertical, ymin=ymin_pspec, ymax=ymax_pspec, color="black", linewidth=2, label="4*sigma*sqrt(ndim)")
        axes_residual.vlines(x=k_vertical, ymin=ymin_residual, ymax=ymax_residual, color="black", linewidth=2)
        axes_pspec.legend()

        print(f"x position of vertical line: {k_vertical}")
        print(f"x-axis limits: {axes_pspec.get_xlim()}")

        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        fig.savefig(self.plot_direc + "/plots/power_spectra.png", dpi=300)


    def plot_panel(self, tick_font_size=7, normalize=False, log=False):
        figsize = (8, 12)
        rows = (self.config_params.iter_num_max // 2) + 1
        cols = 2

        # Load truth field and data
        truth_field_path = f"/npy/truth_field_{self.config_params.z}.npy"
        data_field_path = f"/npy/data_field_{self.config_params.z}.npy"

        if normalize:
            truth_field = np.load(self.plot_direc + truth_field_path)
        else:
            self.truth_field = np.load(self.plot_direc + truth_field_path)
        self.data = np.load(self.plot_direc + data_field_path)

        # Handle 3D data
        if self.config_params.dim == 3:
            truth_field = self.truth_field[10, :, :]
            data = self.data[10, :, :]

        # Create figure and axes
        data_fig, data_axes = plt.subplots(rows, cols, figsize=figsize)
        residuals_fig, residuals_axes = plt.subplots(rows, cols, figsize=figsize)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        k = 0
        for i in range(rows):
            for j in range(cols):
                print("plotting", (i, j))
                self.plot_title = labels[k]

                if i == 0 and j == 0:
                    # Special case for the first plot (truth field)
                    if normalize:
                        hist, bin_edges = np.histogram(truth_field, bins=100)
                        axes[i][j].plot(bin_edges[1:], hist)
                        axes[i][j].set_xscale('symlog')
                        axes[i][j].set_xlabel("Truth field density")
                        axes[i][j].set_xlim(-10**1, 10**2)
                        continue
                    else:
                        im = axes[i][j].imshow(
                            self.truth_field,
                            norm=matplotlib.colors.SymLogNorm(linthresh=0.01) if log else None,
                            vmin=-1, vmax=1
                        )
                        t = axes[i][j].text(20, self.config_params.side_length - 10, "Truth Field", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        self._add_colorbar(axes[i][j], im, tick_font_size, data_fig)

                        # Data field plot
                        im = data_axes[i][j].imshow(
                            self.data,
                            norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1, vmax=jnp.max(self.data)) if log else None,
                            cmap="oranges"
                        )
                        t = data_axes[i][j].text(20, self.config_params.side_length - 10, "Data Field", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        self._add_colorbar(data_axes[i][j], im, tick_font_size, data_fig)

                        # Residuals plot (zeros)
                        im = residuals_axes[i][j].imshow(
                            np.zeros_like(self.truth_field),
                            norm=matplotlib.colors.SymLogNorm(linthresh=0.01) if log else None
                        )
                        t = residuals_axes[i][j].text(20, self.config_params.side_length - 10, "Zero Residuals", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        self._add_colorbar(residuals_axes[i][j], im, tick_font_size, residuals_fig)

                        continue

                # Load the best field for the current iteration
                best_field = np.load(self.plot_direc + f"/npy/best_field_{self.config_params.z}_{k}.npy")
                if self.config_params.dim == 3:
                    best_field = best_field[10, :, :]

                # Plotting based on normalization or not
                if normalize:
                    hist, bin_edges = np.histogram(best_field, bins=1000)
                    axes[i][j].plot(bin_edges[1:], hist)
                    axes[i][j].set_xscale('symlog')
                    axes[i][j].set_xlabel("Density")
                    axes[i][j].set_xlim(-10**1, 10**2)
                else:
                    im = axes[i][j].imshow(
                        best_field,
                        norm=matplotlib.colors.SymLogNorm(linthresh=0.01) if log else None,
                        vmin=-1, vmax=1
                    )
                    self._add_colorbar(axes[i][j], im, tick_font_size, fig)
                    self._add_text(axes[i][j], k, "Iteration", self.plot_title)

                    # Residuals plot
                    residuals = (best_field - self.truth_field) / self.truth_field
                    im = residuals_axes[i][j].imshow(
                        residuals if log else residuals**2,
                        norm=matplotlib.colors.SymLogNorm(linthresh=0.01) if log else None
                    )
                    self._add_colorbar(residuals_axes[i][j], im, tick_font_size, residuals_fig)
                    self._add_text(residuals_axes[i][j], k, "(Current-Truth) Iteration", self.plot_title)

                    # Data panel for current iteration
                    batt_model_instance = Dens2bBatt(best_field, resolution=self.resolution, set_z=self.config_params.z, physical_side_length=self.config_params.physical_side_length, flow=True, free_params=self.config_params.free_params, apply_ska=self.config_params.ska_effects)
                    data = batt_model_instance.temp_brightness
                    im = data_axes[i][j].imshow(
                        data,
                        norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=-1) if log else None
                    )
                    self._add_colorbar(data_axes[i][j], im, tick_font_size, data_fig)
                    self._add_text(data_axes[i][j], k, "Iteration", self.plot_title)

                k += 1
                if k == self.config_params.iter_num_max:
                    break

            # Save the figures
            fig.savefig(self.plot_direc + "/plots/field_panel_iterations.png", dpi=300)
            data_fig.savefig(self.plot_direc + "/plots/data_panel_iterations.png", dpi=300)
            residuals_fig.savefig(self.plot_direc + "/plots/residuals_panel_iterations.png", dpi=300)

            plt.close(data_fig)
            plt.close(fig)
            plt.close(residuals_fig)

    # Helper functions to add colorbar and text
    def _add_colorbar(self, ax, im, tick_font_size, fig):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
        cbar.ax.tick_params(labelsize=tick_font_size)

    def _add_text(self, ax, iteration, title_prefix, title_suffix):
        t = ax.text(20, self.config_params.side_length - 10, f"{title_prefix} #{iteration} \nwith {title_suffix}", color="black", weight='bold')
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))


    # def calc_prior_likelihood(self, field):
    #     field = jnp.reshape(field, self.size)
    #     if self.config_params.dim == 3:
    #         assert np.shape(field) == (self.side_length, self.side_length, self.side_length)
    #     elif self.config_params.dim == 2:
    #         assert np.shape(field) == (self.side_length, self.side_length)
    #     else:
    #         exit()
    #     # note param_init was passed in and is a constant
    #     discrepancy = self.data - self.bias_field(field)
    #     likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
    #     #### prior
    #     if self.new_prior:
    #         counts, power_curr, bin_means = circular_spec_normal(field, self.config_params.num_bins, self.resolution, self.area)
    #         if self.config_params.dim == 2:
    #             counts, power_curr, bin_means = circular_spec_normal(field, self.config_params.num_bins, self.resolution, self.area)
    #         elif self.config_params.dim == 3:
    #             counts, power_curr, bin_means = spherical_p_spec_normal(field, self.config_params.num_bins, self.resolution, self.volume)
    #         # sigma = counts ** 2
    #         mask_high_k = bin_means < 5
    #         x = (self.pspec_box - power_curr).flatten()
    #         prior = np.sum(x[mask_high_k]**2)
    #     elif self.original_prior:
    #         # FT and get only the independent modes
    #         fourier_box = self.fft_jax(field)
    #         fourier_nums_real, fourier_nums_imag = self.independent_only_jax(fourier_box)
    #         real_prior = jnp.dot(fourier_nums_real ** 2, (2 / self.pspec_indep_nums_re))  # Half variance for real
    #         imag_prior = jnp.dot(fourier_nums_imag ** 2, (2 / self.pspec_indep_nums_im))  # Half variance for imag
    #         prior = real_prior + imag_prior
    #     return prior, likelihood

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
        print("labels", labels)
        fig_mse, ax_mse = plt.subplots()
        # ax_mse.semilogy(mse_vals, label="Mean square error", ls=None)
        ax_mse.semilogy(prior_arr, label="Prior", ls="", marker="o", alpha=0.4)
        ax_mse.semilogy(like_arr, label="Likelihood", ls="", marker="o", alpha=0.4)
        ax_mse.semilogy(posterior_arr, label="Posterior", ls="", marker="o", alpha=0.4)

        print("-" * 40)  # Prints a line divider
        print("prior_arr")
        print(prior_arr)
        print("-" * 40)

        print("-" * 40)  # Prints a line divider
        print("like_arr")
        print(like_arr)
        print("-" * 40)

        print("-" * 40)  # Prints a line divider
        print("posterior_arr")
        print(posterior_arr)
        print("-" * 40)

        # ax_mse.semilogy(final_function_value_output, label="Output from minimizer", ls="--")
        # Plot NaN values separately with 'X' marker
        # ax_mse.semilogy(x_vals[np.isnan(prior_arr)], np.ones_like(x_vals[np.isnan(prior_arr)]), marker="x", ls="",
        #                 color="C0", label="Prior (NaN)")
        # ax_mse.semilogy(x_vals[np.isnan(like_arr)], np.ones_like(x_vals[np.isnan(like_arr)]), marker="x", ls="",
        #                 color="C1", label="Likelihood (NaN)")
        # ax_mse.semilogy(x_vals[np.isnan(posterior_arr)], np.ones_like(x_vals[np.isnan(posterior_arr)]), marker="x",
        #                 ls="", color="C2", label="Posterior (NaN)")

        # for iter_num_big in range(self.config_params.iter_num_max):
        #     if labels[iter_num_big] == "prior_off" and iter_num_big == 0:
        #         plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="*", c="black", label="likelihood only, mask off")
        #     elif labels[iter_num_big] == "likelihood_off" and iter_num_big == 1:
        #         plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="o", c="black", label="prior only, mask on")
        #     elif labels[iter_num_big] == "prior_off":
        #         plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="*", c="black")
        #     elif labels[iter_num_big] == "likelihood_off":
        #         plt.semilogy(iter_num_big, mse_vals[iter_num_big], marker="o", c="black")
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

    def plot_5_panel(self, shared_axis=True, slice_idx=10):
        print("self.size[-1]")
        print(self.size[-1])
        cube_height = self.size[-1]
        if cube_height < 20:
            for i in range(cube_height):
                self.plot_5_panel_slice(slice_idx=i, shared_axis=shared_axis)
        else:
            self.plot_5_panel_slice(slice_idx=slice_idx, shared_axis=shared_axis)
            # self.plot_5_panel_slice(slice_idx=slice_idx, shared_axis=False)


    def plot_5_panel_slice(self, slice_idx, shared_axis=False):
        """This method gives a nice five panel plot showing the data, starting field, predicted field, truth field, residual"""

        assert np.shape(self.truth_field) == np.shape(self.data)
        assert np.shape(self.best_field_reshaped) == np.shape(self.data)

        residual = self.truth_field - self.best_field_reshaped

        # norm_shared = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min(np.min(self.best_field_reshaped), np.min(self.truth_field)),
        #                                vmax=max(np.max(self.best_field_reshaped), np.max(self.truth_field)))

        min_density = min(np.min(self.best_field_reshaped), np.min(self.truth_field))
        max_density = max(np.max(self.best_field_reshaped), np.max(self.truth_field))

        norm_shared_residual = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=np.min(residual),
                                       vmax=np.max(residual))

        fig, (ax1, ax6, ax5, ax2, ax3, ax4) = plt.subplots(1, 6, figsize=(20, 5))

        # Check if fields are 3D or 2D
        if self.data.ndim == 3:
            data_slice = self.data[slice_idx, :, :]
            best_field_slice = self.best_field_reshaped[slice_idx, :, :]
            truth_field_slice = self.truth_field[slice_idx, :, :]
            s_field_original = jnp.reshape(self.s_field_original, (self.size))[slice_idx, :, :]
        else:
            # If fields are 2D, no slicing needed
            data_slice = self.data
            best_field_slice = self.best_field_reshaped
            truth_field_slice = self.truth_field
            s_field_original = jnp.reshape(self.s_field_original, (self.size))


        # Calculate the residual (truth - best)
        residual_slice = truth_field_slice - best_field_slice

        # Plotting Observed data
        im1 = ax1.imshow(data_slice, cmap="plasma")
        ax1.set_title("Observed Data") # + (" (Slice {})".format(slice_idx) if self.data.ndim == 3 else ""))
        ax1.set_xlabel("Pixel #")

        kurt_best = kurtosis(self.best_field_reshaped.flatten(), fisher=True)
        kurt_truth = kurtosis(self.truth_field.flatten(), fisher=True)

        skew_best = skew(self.best_field_reshaped.flatten())
        skew_truth = skew(self.truth_field.flatten())

        print(f"Best: skew = {skew_best:.3e}, kurt = {kurt_best:.3e}")
        print(f"Truth: skew = {skew_truth:.3e}, kurt = {kurt_truth:.3e}")

        # Plotting Inferred density
        if shared_axis:
            im2 = ax2.imshow(best_field_slice, vmin=min_density, vmax=max_density)
            ax2.set_aspect("equal")

        else:
            im2 = ax2.imshow(best_field_slice) #, norm=norm_shared)

        title = f"Inferred Density" # skew = {skew_best:.2e}, kurt = {kurt_best:.2e}"
        if self.data.ndim == 3:
            title += f" (Slice {slice_idx})"
        ax2.set_title(title)

        ax2.set_xlabel("Pixel #")

        # Plotting Truth
        if shared_axis:
            im3 = ax3.imshow(truth_field_slice, vmin=min_density, vmax=max_density)
            ax3.set_aspect("equal")

        else:
            im3 = ax3.imshow(truth_field_slice)

        title = f"Truth" # skew = {skew_truth:.2e}, kurt = {kurt_truth:.2e}"
        if self.truth_field.ndim == 3:
            title += f" (Slice {slice_idx})"
        ax3.set_title(title)

        ax3.set_xlabel("Pixel #")

        # Plotting Residual (truth-best)
        im4 = ax4.imshow(residual_slice) #, norm=norm_shared_residual)
        ax4.set_title("Residual") #+ (" (Slice {})".format(slice_idx) if self.truth_field.ndim == 3 else ""))
        ax4.set_xlabel("Pixel #")

        # Plotting Starting field (truth-best)
        im5 = ax5.imshow(s_field_original) #, norm=norm_shared_residual)
        ax5.set_title("Starting Field") #+ (" (Slice {})".format(slice_idx) if self.truth_field.ndim == 3 else ""))
        ax5.set_xlabel("Pixel #")

        # Plotting Starting field (truth-best)
        best_data_slice_inferred = Dens2bBatt(self.best_field_reshaped, resolution=self.resolution, set_z=self.config_params.z, physical_side_length=self.config_params.physical_side_length, flow=True, free_params=self.config_params.free_params, apply_ska=self.config_params.ska_effects)

        if self.config_params.dim == 3:
            im6 = ax6.imshow(best_data_slice_inferred.temp_brightness[slice_idx, :, :])  # , norm=norm_shared_residual)
        elif self.config_params.dim == 2:
            im6 = ax6.imshow(best_data_slice_inferred.temp_brightness)  # , norm=norm_shared_residual)

        else:
            print("Dimension not supported")
            exit()

        ax6.set_title("Inferred Data")  # + (" (Slice {})".format(slice_idx) if self.truth_field.ndim == 3 else ""))
        ax6.set_xlabel("Pixel #")

        # Create individual colorbars
        # Adding colorbars with consistent formatting
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
        cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
        cbar5 = fig.colorbar(im5, ax=ax5, orientation='vertical', fraction=0.046, pad=0.04)
        cbar6 = fig.colorbar(im6, ax=ax6, orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to avoid overlap
        fig.subplots_adjust(wspace=0.3)  # Adjust space between subplots
        fig.tight_layout()
        ## getting plot title
        if self.config_params.noise_off:
            save_plot_title = f"5_panel_z_{self.config_params.z}_no_noise_.png"
        else:
            save_plot_title = f"5_panel_z_{self.config_params.z}_w_noise_.png"

        if shared_axis:
            save_plot_title = f"shared_axis_{save_plot_title}"

        fig.savefig(f"{self.plot_direc}/plots/{save_plot_title}", dpi=300)

        print("Saving five panel plot...")

        plt.close()

    def plot_3_panel(self):
        """This method gives a nice three panel plot showing the data, predicted field, and truth field"""
        print("MINIMUM VALUE OF DATA FIELD")
        print(jnp.min(self.data))
        plt.close()
        plt.hist(self.data.flatten(), bins=100, density=True, histtype="stepfilled")
        plt.title("21cm values")
        plt.savefig(f"{self.plot_direc}/plots/data_hist.png")
        # Normalize the color range for Inferred density and Truth
        # norm_shared = colors.Normalize(vmin=min(np.min(self.best_field_reshaped), np.min(self.truth_field)),
        #                                vmax=max(np.max(self.best_field_reshaped), np.max(self.truth_field)))

        assert np.shape(self.truth_field) == np.shape(self.data)
        assert np.shape(self.best_field_reshaped) == np.shape(self.data)

        residual = self.truth_field - self.best_field_reshaped

        # norm_shared = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=min(np.min(self.best_field_reshaped), np.min(self.truth_field)),
        #                                vmax=max(np.max(self.best_field_reshaped), np.max(self.truth_field)))

        norm_shared_residual = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=np.min(residual),
                                       vmax=np.max(residual))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

        # Check if fields are 3D or 2D
        if self.data.ndim == 3:
            slice_idx = 10  # You can change this to the desired slice
            data_slice = self.data[slice_idx, :, :]
            best_field_slice = self.best_field_reshaped[slice_idx, :, :]
            truth_field_slice = self.truth_field[slice_idx, :, :]
        else:
            # If fields are 2D, no slicing needed
            data_slice = self.data
            best_field_slice = self.best_field_reshaped
            truth_field_slice = self.truth_field

        # Calculate the residual (truth - best)
        residual_slice = truth_field_slice - best_field_slice

        # Plotting Observed data
        im1 = ax1.imshow(data_slice, cmap="plasma")
        ax1.set_title("Observed Data") # + (" (Slice {})".format(slice_idx) if self.data.ndim == 3 else ""))
        ax1.set_xlabel("Pixel #")

        kurt_best = kurtosis(self.best_field_reshaped.flatten(), fisher=True)
        kurt_truth = kurtosis(self.truth_field.flatten(), fisher=True)

        skew_best = skew(self.best_field_reshaped.flatten())
        skew_truth = skew(self.truth_field.flatten())


        # Plotting Inferred density
        im2 = ax2.imshow(best_field_slice)
        ax2.set_title(f"Inferred Density") # f"skew = {skew_best:.2e}, kurt = {kurt_best:.2e}) " + f"(Slice {slice_idx})" if self.data.ndim == 3 else "")
        ax2.set_xlabel("Pixel #")

        # Plotting Truth
        im3 = ax3.imshow(truth_field_slice)
        ax3.set_title(f"Truth") # + f"skew = {skew_truth:.2e}, kurt = {krmurt_truth:.2e}) " + f"(Slice {slice_idx})" if self.truth_field.ndim == 3 else "")
        ax3.set_xlabel("Pixel #")

        # Plotting Residual (truth-best)

        im4 = ax4.imshow(residual_slice) #, norm=norm_shared_residual)
        ax4.set_title("Residual") # + "(Slice {})".format(slice_idx) if self.truth_field.ndim == 3 else "")
        ax4.set_xlabel("Pixel #")
        # Create individual colorbars
        # Adding colorbars with consistent formatting
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
        cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to avoid overlap
        fig.subplots_adjust(wspace=0.3)  # Adjust space between subplots
        fig.tight_layout()

        ## getting plot title
        if self.config_params.noise_off:
            save_plot_title = f"3_panel_z_{self.config_params.z}_no_noise_.png"
        else:
            save_plot_title = f"3_panel_z_{self.config_params.z}_w_noise_.png"


        fig.savefig(f"{self.plot_direc}/plots/{save_plot_title}", dpi=300)

        print("Saving three panel plot...")

        plt.close()



    def check_field(self, field, title, normalize=False, save=True, show=False, iteration_number=-1):
        # plotting
        if self.config_params.dim == 3:
            # take slice of field
            field = field[10, :, :]
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
        # plt.yscale('symlog')

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
        batt_model_instance = Dens2bBatt(self.truth_field, resolution=self.resolution, set_z=self.config_params.z, physical_side_length=self.config_params.physical_side_length, flow=True, free_params=self.config_params.free_params, apply_ska=self.config_params.ska_effects)

        x_HII = 1 - batt_model_instance.X_HI
        plt.scatter(x_HII, residual, color="black", s= 8)
        plt.ylabel("Residual (truth - inferred)")
        plt.xlabel("X_HII")
        plt.savefig(f"{self.plot_direc}/plots/residual_by_frac.png")


class ConfigParam:
    def __init__(self, ska_effects, free_params, z, truth_field, brightness_temperature_field, num_bins, nothing_off, plot_direc, side_length, physical_side_length,
                 dimensions=2, iter_num_max=10, rest_num_max=3, noise_off=False,
                 run_optimizer=False, mse_plot_on=False,
                 weighted_prior=None, new_prior=False, old_prior=False, verbose=False,
                 debug=False, use_truth_mm=False, save_prior_likelihood_arr=False, seed=1010,
                 create_instance=False, use_matter_pspec_starting_field=False, normalize_everything=False,
                 cov_matrix_data=True, know_neutral_pixels=False, ionized_threshold=None):
        """
        :param free_params - dictionary with free parameters to use
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
        :param use_matter_pspec_starting_field (Default: False) - start field with a power spectrum matching truth from CAMB
        :param normalize_everything - normalize the entire field
        :param cov_matrix_data - use cov matrix, can only work in 2D because of memory issues
        :param know_neutral_pixels - assume you know neutral pixels through analytic inversion of Battaglia+2013 model

        """
        self.ska_effects = ska_effects
        self.free_params = free_params
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
        self.use_matter_pspec_starting_field = use_matter_pspec_starting_field
        self.normalize_everything = normalize_everything
        self.cov_matrix_data = cov_matrix_data
        self.know_neutral_pixels = know_neutral_pixels
        self.ionized_threshold = ionized_threshold
        if self.cov_matrix_data:
            assert self.dim == 2
        assert(self.new_prior != self.old_prior)

    def save_to_file(self, directory, filename="config_run.txt"):
        with open(directory + "/" + filename, "w") as file:
            for key, value in self.__dict__.items():
                file.write(f"{key}: {value}\n")

# Print the result to verify
# print(f'SKA Effects Enabled: {ska_effects}')

k_0_fiducial = 0.185 * 0.676 # changing from Mpc/h to Mpc
alpha_fiducial = 0.564
b_0_fiducial = 0.593
midpoint_z_fiducial = 7
tan_fiducial = 2

static_params = {"b_0": b_0_fiducial, "alpha": alpha_fiducial, "k_0": k_0_fiducial, "tanh_slope": tan_fiducial,
                   "avg_z": midpoint_z_fiducial}  # b_0=0.5, alpha=0.2, k_0=0.1)

def grid_test(free_param_name, free_param_arr, static_redshift=True,
              ska_effects=False,
              nominal=False, side_length=16, physical_side_length=16, bins=8, brightness_temperature_field=None,
              truth_field=None, iter_num_max=3, rest_num_max=2, nothing_off=False, dimensions=3, cov_matrix_data=False):
    import GPUtil
    import gc
    fiducial_params = static_params.copy()
    # try to stop this from crashing
    if nominal:
        z = fiducial_params["redshift_run"]

        # params = ConfigParam(ska_effects=ska_effects, free_params=fiducial_params, z=z, truth_field=truth_field,  brightness_temperature_field=brightness_temperature_field, num_bins=bins,
        #                      nothing_off=nothing_off, plot_direc="", side_length=side_length, physical_side_length=physical_side_length,
        #                      dimensions=3, iter_num_max=iter_num_max, rest_num_max=rest_num_max, noise_off=True,
        #                      run_optimizer=True, mse_plot_on=False,
        #                      weighted_prior=None, new_prior=True, old_prior=False, verbose=False,
        #                      debug=False, use_truth_mm=True, save_prior_likelihood_arr=False, seed=1,
        #                      create_instance=False)

    else:
        for free_param in free_param_arr:
            gpu = GPUtil.getGPUs()[0]
            print(
                f"GPU ID: {gpu.id}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")
            if free_param_arr == [None]:  # no free parameters
                z = fiducial_params["redshift_run"]
            else:
                free_param = np.round(free_param, decimals=4)
                if static_redshift:
                    fiducial_params[free_param_name] = free_param # update fiducial param list with varying parameter
                    z = fiducial_params["redshift_run"]
                else: # varying redshift
                    z = free_param
    params = ConfigParam(ska_effects=ska_effects, free_params=fiducial_params, z=z, truth_field=truth_field,
                         brightness_temperature_field=brightness_temperature_field, num_bins=bins,
                         nothing_off=nothing_off, plot_direc="", side_length=side_length,
                         physical_side_length=physical_side_length,
                         dimensions=dimensions, iter_num_max=iter_num_max, rest_num_max=rest_num_max, noise_off=True,
                         run_optimizer=True, mse_plot_on=False,
                         weighted_prior=None, new_prior=True, old_prior=False, verbose=False,
                         debug=False, use_truth_mm=False, save_prior_likelihood_arr=True, seed=1,
                         create_instance=False, use_matter_pspec_starting_field=False, normalize_everything=False,
                         cov_matrix_data=cov_matrix_data)


    # DO FOR BOTH
    samp = InferDens(params, s_field=None) # setting s-field to none for first iteration
    samp.make_ionisation_level_residual_plot()
    samp.plot_5_panel()
    samp.plot_3_panel()
    samp.plot_mask()
    samp.plot_pspecs()
    samp.plot_all_optimizer_vals()
    gc.collect()



if __name__ == "__main__":
    run_optimizer = True
    # for z in [10, 9, 8, 7, 6.5]:
    for z in [12]:
        static_params["redshift_run"] = z
        # truth_field = np.load(f"21cmfast_fields_1Mpcpp/density_{z}.npy")
        # brightness_temperature_field = np.load(f"21cmfast_fields_1Mpcpp/brightness_temp_{z}.npy")
        # that one doesn't work
        grid_test("none", [None], static_redshift=True, ska_effects=ska_effects, nominal=True,
                  bins=16,
                  side_length=32,
                  physical_side_length=32,
                  iter_num_max=10,
                  rest_num_max=1,
                  nothing_off=True,
                  truth_field=[], brightness_temperature_field=[], dimensions=2, cov_matrix_data=True)
    exit()
    # Define parameter ranges
    # alpha_values = np.linspace(0.1, 2, 10)
    # b_0_values = np.logspace(-2, 2, 10)
    # k_0_values = np.logspace(-2, 2, 10)
    # avg_z_values = np.linspace(5, 8, 10)

    # Create a mesh grid
    # alpha_grid, b_0_grid, k_0_grid, avg_z_grid = np.meshgrid(alphas, b_0s, k_0s, avg_z, indexing='ij')

    # Reshape to create a list of all combinations
    # alpha_values = alpha_grid.flatten()
    # b_0_values = b_0_grid.flatten()
    # k_0_values = k_0_grid.flatten()
    # avg_z_values = avg_z_grid.flatten()

    # Number of parameter combinations
    # num_combinations = alpha_values.size
    # Iterate over each combination of parameters
    
    for z in [11]:
        for i in range(len(alpha_values)):
            static_params["redshift_run"] = z
            alpha = alpha_values[i]
            b_0 = b_0_values[i]
            k_0 = k_0_values[i]
            avg_z = avg_z_values[i]
            truth_field = np.load(f"/fred/oz113/sberger/paper_1_density/Grad2Dens/src/21cmfast_fields/density_{z}.npy")
            brightness_temperature_field = np.load(f"/fred/oz113/sberger/paper_1_density/Grad2Dens/src/21cmfast_fields/brightness_temp_{z}.npy")
            print(np.shape(truth_field))
            grid_test("alpha", [alpha], static_redshift=True, ska_effects=ska_effects, truth_field=truth_field, brightness_temperature_field=brightness_temperature_field)
            grid_test("b_0", [b_0], static_redshift=True, ska_effects=ska_effects, truth_field=truth_field, brightness_temperature_field=brightness_temperature_field)
            grid_test("k_0", [k_0], static_redshift=True, ska_effects=ska_effects, truth_field=truth_field, brightness_temperature_field=brightness_temperature_field)
            grid_test("avg_z", [avg_z], static_redshift=True, ska_effects=ska_effects, truth_field=truth_field, brightness_temperature_field=brightness_temperature_field)

        # for i in range(num_combinations_start, num_combinations):
        #     print(f"Completed {i} out of {num_combinations} for z = {z}!")
        #     alpha = alpha_values[i]
        #     b_0 = b_0_values[i]
        #     k_0 = k_0_values[i]
        #     avg_z = avg_z_values[i]
        # 
        #     # Call your `grid_test` function with these parameters
        #     grid_test("alpha", [alpha], static_redshift=True, ska_effects=ska_effects)
        #     grid_test("b_0", [b_0], static_redshift=True, ska_effects=ska_effects)
        #     grid_test("k_0", [k_0], static_redshift=True, ska_effects=ska_effects)
        #     grid_test("avg_z", [avg_z], static_redshift=True, ska_effects=ska_effects)
