"""
Infrastructure to generate MAPs of the matter density field
Created November, 2022
Written by Sabrina Berger
"""

from jax_main import SwitchMinimizer
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
import os
from matplotlib.ticker import MaxNLocator
from jax_battaglia_full import Dens2bBatt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    # TODO change to take input config file
    def __init__(self, seed, z, truth_field,  brightness_temperature_field, num_bins=None, mask_ionized=False, nothing_off=False, plot_direc="", side_length=256,
                 dimensions=2, iter_num_max=10, rest_num_max=3, noise_off=False, save_posterior_values=False, run_optimizer=False,
                 pspec_on_plot=False, mse_plot_on=False, weighted_prior=None, new_prior=False, old_prior=False, verbose=False, debug=False, use_truth_mm=False):
        self.z = z
        self.seed = seed
        super().__init__(self.seed, self.z, num_bins=num_bins, data=brightness_temperature_field, side_length=side_length,
                                   dimensions=dimensions, plot_direc=plot_direc,
                                    verbose=verbose, noise_off=noise_off, weighted_prior=weighted_prior, new_prior=new_prior,
                                    old_prior=old_prior, truth_field=truth_field, debug=debug, use_truth_mm=use_truth_mm)
        self.iter_num_max = iter_num_max # number of iterations over which to minimize posterior
        self.rest_num_max = rest_num_max # number of iterations to rest on the given configuration
        self.save_posterior_values = save_posterior_values

        self.pspec_on = pspec_on_plot
        self.mse_plot_on = mse_plot_on
        self.mask_ionized = mask_ionized
        self.nothing_off = nothing_off

        self.final_function_value_output = jnp.zeros(self.iter_num_max)
        self.mse_vals = jnp.empty(self.iter_num_max)
        self.posterior_vals = jnp.empty(self.iter_num_max)
        self.likelihood_vals = jnp.empty(self.iter_num_max)
        self.prior_vals = jnp.empty(self.iter_num_max)
        self.pixel_tracks_ionized = jnp.empty(self.iter_num_max)
        self.pixel_tracks_neutral = jnp.empty(self.iter_num_max)
        self.hessian_vals = jnp.empty(self.iter_num_max)

        self.labels = []

        if plot_direc == "":
            perc_ionized = round(len(self.ionized_indices) / self.side_length**2, 1)
            if old_prior:
                new_direc = f"z_{self.z}_perc_ionized_{perc_ionized}_old_seed_{seed}_bins_{num_bins}"
            elif new_prior:
                new_direc = f"z_{self.z}_perc_ionized_{perc_ionized}_seed_{seed}_bins_{num_bins}"

            if self.truth_field.any() != None:
                new_direc = f"z_{self.z}_diff_start_" + new_direc
            try:
                os.mkdir(new_direc)
                os.mkdir(new_direc + "/plots")
                os.mkdir(new_direc + "/npy")
            except:
                print("directory already exists")
            self.plot_direc = new_direc
        else:
            self.plot_direc = plot_direc # this overwrites the plot_directory in the SwitchMinimizer class

        if run_optimizer:
            self.infer_density_field()
        print("Saving plots here...")
        print(self.plot_direc)

    def check_threshold(self):
        batt_model_instance = Dens2bBatt(self.best_field_reshaped, delta_pos=1, set_z=self.z, flow=True)
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
        for iter_num_big in range(self.iter_num_max):
            self.iter_num_big = iter_num_big
            if iter_num_big == 0:
                if self.nothing_off:
                    self.prior_off = False
                    self.likelihood_off = False
                else:
                    # first iteration, we keep just the likelihood on
                    self.prior_off = True
                    self.likelihood_off = False
                self.run_grad_descent()

                self.check_field(self.s_field_original, "starting field", show=True, save=True,
                                 iteration_number=-1)
                self.check_field(self.truth_field, "truth_field", show=True, save=True,
                                 iteration_number=-1)
                self.check_field(self.data, "data", show=True, save=True,
                                 iteration_number=-1)
                np.save(self.plot_direc + f"/npy/truth_field_{self.z}.npy", self.truth_field)
                np.save(self.plot_direc + f"/npy/data_field_{self.z}.npy", self.data)

            else:
                if not self.nothing_off: # if both on this is skipped
                    if rest_num == self.rest_num_max or self.prior_off:
                        self.prior_off = not self.prior_off
                        self.likelihood_off = not self.likelihood_off
                        rest_num = 0
                    if self.likelihood_off:
                        self.mask_ionized = True
                self.rerun(likelihood_off=self.likelihood_off, prior_off=self.prior_off, mask_ionized=self.mask_ionized)

            print("--------------------------------------------------------")
            print(f"Iteration #{iter_num_big}")
            if self.verbose:
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
            self.check_field(self.best_field_reshaped, self.plot_title, show=True, save=True,
                                 iteration_number=iter_num_big)
            np.save(self.plot_direc + f"/npy/best_field_{self.z}_{iter_num_big}.npy", self.best_field_reshaped)
            self.labels.append(self.plot_title) # saving configuration of each iteration in list
            np.save(self.plot_direc + f"/npy/labels.npy", self.labels)
            rest_num += 1 # INCREASE REST NUM ONE

            # saving information only at the end of a round of likelihood/prior off
            # getting 0th ionized/neutral index and saving to array
            ionized_ind = self.ionized_indices[0]
            neutral_ind = self.neutral_indices[0]
            ionized_val = self.best_field.flatten()[ionized_ind]
            neutral_val = self.best_field.flatten()[neutral_ind]
            self.pixel_tracks_ionized = self.pixel_tracks_ionized.at[iter_num_big].set(ionized_val)
            self.pixel_tracks_neutral = self.pixel_tracks_neutral.at[iter_num_big].set(neutral_val)

            # getting mse vals, final function val from optimizer, likelihood vals, posterior vals
            self.mse_vals = self.mse_vals.at[iter_num_big].set(self.check_threshold())
            self.final_function_value_output = self.final_function_value_output.at[iter_num_big].set(self.final_func_val)

            # prior, likelihood = self.calc_prior_likelihood(self.best_field_reshaped)
            # self.prior_vals = self.prior_vals.at[iter_num_big].set(prior)
            # self.likelihood_vals = self.likelihood_vals.at[iter_num_big].set(likelihood)
            # self.posterior_vals = self.likelihood_vals.at[iter_num_big].set(prior + likelihood)

            # # save intermediate arrays of all important quantities
            # np.save(self.plot_direc + f"/npy/prior_vals_{self.z}.npy", self.prior_vals)
            # np.save(self.plot_direc + f"/npy/likelihood_vals_{self.z}.npy", self.likelihood_vals)
            # np.save(self.plot_direc + f"/npy/posterior_vals_{self.z}.npy", self.posterior_vals)
            # np.save(self.plot_direc + f"/npy/optimizer_output_vals_{self.z}.npy", self.final_function_value_output)
            # np.save(self.plot_direc + f"/npy/hessian_vals_{self.z}.npy", self.hessian_vals)
            # np.save(self.plot_direc + f"/npy/mse_vals_{self.z}.npy", self.mse_vals)

        np.save(self.plot_direc + f"/npy/best_field_{self.z}_FINAL.npy",
                self.best_field_reshaped)

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
        truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
        self.num_k_modes = self.side_length
        self.resolution = self.side_length / self.num_k_modes  # actual length / number of pixels
        self.area = self.side_length ** 2
        _, pspec_truth, kvals = self.p_spec_normal(truth_field, self.num_bins, self.resolution, self.area)
        fig_pspec, axes_pspec = plt.subplots()
        axes_pspec.loglog(kvals, pspec_truth, label=f"Truth field", lw=3, c="k")
        for i, k in enumerate([0, 2, 3, 4, 5]):
            best_field = np.load(self.plot_direc + f"/npy/best_field_{self.z}_{k}.npy")
            counts, pspec_normal, kvals = self.p_spec_normal(best_field, self.num_bins, self.resolution, self.area)
            # counts, pspec_pre, kvals = self.p_spec_pre(best_field, self.num_bins)
            # poisson_rms = np.sqrt(np.abs(pspec_normal - pspec_pre)) / np.sqrt(counts)
            # print(poisson_rms)
            # axes_pspec.errorbar(kvals, pspec_normal, yerr=poisson_rms, label=f"Iteration # {k} with {title}", ls="--", alpha=0.3)
            axes_pspec.loglog(kvals, pspec_normal,
                              label=f"Iteration # {k} with {labels[k]}")

        axes_pspec.set_xscale("log")
        axes_pspec.set_yscale("log")
        axes_pspec.legend()
        axes_pspec.set_ylabel("Power [units]")
        axes_pspec.set_xlabel("k [units]")
        fig_pspec.savefig(self.plot_direc + "/plots/pspec.png", dpi=300)
        plt.close(fig_pspec)

    def plot_panel(self, tick_font_size=7, normalize=False):
        figsize = (8, 12)
        rows = 5
        cols = 2

        if normalize:
            truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
        else:
            truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
            data = np.load(self.plot_direc + f"/npy/data_field_{self.z}.npy")
            data_fig, data_axes = plt.subplots(rows, cols, figsize=figsize)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        k = 0
        for i in range(rows):
            for j in range(cols):
                print("plotting", (i,j))
                self.plot_title = labels[k]
                if i == 0 and j == 0:
                    if normalize:
                        hist, bin_edges = np.histogram(truth_field, bins=1000)
                        axes[i][j].plot(bin_edges[1:], hist)
                        axes[i][j].set_xscale('symlog')
                        axes[i][j].set_xlabel("truth field density")
                        axes[i][j].set_xlim(-10**1, 10**2)
                        continue
                    else:
                        im = axes[i][j].imshow(truth_field, norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=0, vmax=1e-1))#vmin=-, vmax=np.max(self.truth_field)))
                        divider = make_axes_locatable(axes[i][j])
                        cax = divider.append_axes('bottom', size='5%', pad=0.05)
                        cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                        cbar.ax.tick_params(labelsize=tick_font_size)
                        t = axes[i][j].text(20, 220, f"Truth Field", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        axes[i][j].set_xticks([])
                        axes[i][j].set_aspect('equal')

                        im = data_axes[i][j].imshow(data)
                        divider = make_axes_locatable(data_axes[i][j])
                        cax = divider.append_axes('bottom', size='5%', pad=0.05)
                        cbar = data_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                        cbar.ax.tick_params(labelsize=tick_font_size)
                        t = data_axes[i][j].text(20, 220, f"Truth Data", color="black", weight='bold')
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        data_axes[i][j].set_xticks([])
                        data_axes[i][j].set_aspect('equal')
                        continue

                best_field = np.load(self.plot_direc + f"/npy/best_field_{self.z}_{k}.npy")
                if normalize:
                    hist, bin_edges = np.histogram(best_field, bins=1000)
                    axes[i][j].plot(bin_edges[1:], hist)
                    axes[i][j].set_xscale('symlog')
                    axes[i][j].set_xlabel("density")
                    axes[i][j].set_xlim(-10 **1, 10 ** 2)

                else:
                    im = axes[i][j].imshow(best_field,norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=0, vmax=1e-1))
                    divider = make_axes_locatable(axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = axes[i][j].text(20, 220, f"Iteration #{k} \nwith " + self.plot_title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    axes[i][j].set_xticks([])
                    axes[i][j].set_aspect('equal')

                    batt_model_instance = Dens2bBatt(best_field, delta_pos=1, set_z=self.z, flow=True)
                    data = batt_model_instance.temp_brightness
                    im = data_axes[i][j].imshow(data)
                    divider = make_axes_locatable(data_axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = data_fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = data_axes[i][j].text(20, 220, f"Iteration #{k} \nwith " + self.plot_title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    data_axes[i][j].set_xticks([])
                    data_axes[i][j].set_aspect('equal')

                k += 1
            if normalize:
                fig.savefig(self.plot_direc + "/plots/normalize_field_panel_iterations.png", dpi=300)
            else:
                fig.savefig(self.plot_direc + "/plots/field_panel_iterations.png") #, dpi=300)
                data_fig.tight_layout()
                data_fig.savefig(self.plot_direc + "/plots/data_panel_iterations.png", dpi=300)

                plt.close(data_fig)
            plt.close(fig)

    def calc_prior_likelihood(self, field):
        field = jnp.reshape(field, self.size)
        assert np.shape(field) == (self.side_length, self.side_length)
        # note param_init was passed in and is a constant
        discrepancy = self.data - self.bias_field(field)
        likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
        #### prior
        if self.new_prior:
            counts, power_curr, _ = self.p_spec_normal(field, self.num_bins)
            sigma = counts ** 2
            x = (self.pspec_box - power_curr).flatten()
            prior = jnp.dot(x ** 2, 1 / sigma)
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
        mse_vals = np.load(self.plot_direc + f"/npy/mse_vals_{self.z}.npy")
        prior_arr = np.load(self.plot_direc + f"/npy/prior_vals_{self.z}.npy")
        like_arr = np.load(self.plot_direc + f"/npy/likelihood_vals_{self.z}.npy")
        posterior_arr = np.load(self.plot_direc + f"/npy/posterior_vals_{self.z}.npy")
        print("like arr")
        print(like_arr)
        print("prior arr")
        print(prior_arr)
        print("actual")
        print(posterior_arr)
        print("summed")
        print(like_arr + prior_arr)
        print("actual")
        print(posterior_arr)
        final_function_value_output = np.load(self.plot_direc + f"/npy/optimizer_output_vals_{self.z}.npy")
        labels = np.load(self.plot_direc + f"/npy/labels.npy")
        fig_mse, ax_mse = plt.subplots()
        ax_mse.semilogy(mse_vals, label="Mean square error", ls=None)
        ax_mse.semilogy(prior_arr, label="Prior", ls="--")
        ax_mse.semilogy(like_arr, label="Likelihood", ls="--")
        # ax_mse.semilogy(posterior_arr, label="Posterior")
        # ax_mse.semilogy(final_function_value_output, label="Output from minimizer", ls="--")

        for iter_num_big in range(self.iter_num_max):
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
        fig_mse.savefig(self.plot_direc + f"/plots/prior_likelihood_mse_post_{self.z}.png", dpi=300)

    def plot_2_mse(self, mse_val_1="data_adversarial_new_prior", mse_val_2="data_adversarial_classic_prior"):
        ### plotting mse vals
        mse_vals_1 = np.load(mse_val_1 + f"/npy/mse_vals_{self.z}.npy")
        mse_vals_2 = np.load(mse_val_2 + f"/npy/mse_vals_{self.z}.npy")

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
        fig_mse.savefig(f"compare_mse_post_{self.z}.png")

    def plot_3_panel(self):
        """This method gives a nice three panel plot showing the data, predicted field, and truth field"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        im1 = ax1.imshow(self.data)
        ax1.set_title("Observed data")
        im2 = ax2.imshow(self.opt_result)
        ax2.set_title("Inferred density")
        im3 = ax3.imshow(self.truth_field)
        ax3.set_title("Truth")
        fig.tight_layout()
        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        fig.colorbar(im3, ax=ax3)
        ## getting plot title
        if self.noise_off:
            fig.savefig(f"{self.plot_direc}/plots/3_panel_z_{self.z}_no_noise_.png", dpi=300)
        else:
            fig.savefig(f"{self.plot_direc}/plots/3_panel_z_{self.z}_w_noise.png", dpi=300)
        plt.close()

    def check_field(self, field, title, normalize=False, save=True, show=False, iteration_number=-1):
        # plotting
        fig, axes = plt.subplots()
        if normalize:
            field = (field - jnp.min(field)) / (jnp.max(field) - jnp.min(field))
        if field.ndim < 2:
            axes.plot(field)
        else:
            im = axes.imshow(field, norm=matplotlib.colors.SymLogNorm(linthresh=0.01, vmin=np.min(field), vmax=np.max(field)))
            fig.colorbar(im, ax=axes)
            # plt.clim(-1, 16)
        if iteration_number >= 0:
            t = axes.text(20, 240, f"Iteration #{iteration_number} with " + title)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        else:
            t = axes.text(20, 240, "z = " + str(self.z) + " " + title)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

        fig.tight_layout()
        if save:
            if iteration_number >= 0:
                fig.savefig(f"{self.plot_direc}/plots/" + f"iter_num_{iteration_number}_" + f"{self.z}" + "_battaglia.png")
            else:
                fig.savefig(f"{self.plot_direc}/plots/" + f"{title}_{self.z}" + "_battaglia.png")

        if show:
            fig.show()
        plt.close(fig)

if __name__ == "__main__":
    print("starting")
    # samp = InferDens(z=8, num_bins=256, nothing_off=False, mask_ionized=True, iter_num_max=10, plot_direc="data_adversarial_new_prior_256_bins", run_optimizer=True, weighted_prior=False)
    # samp = InferDens(z=8, num_bins=10, nothing_off=False, mask_ionized=True, iter_num_max=10, plot_direc="data_adversarial_new_prior_10_bins", run_optimizer=False, weighted_prior=False)
    # samp.plot_mse_prior_likelihood()
    # samp = InferDens(z=8, num_bins=100, nothing_off=False, mask_ionized=True, iter_num_max=10, plot_direc="data_adversarial_new_prior_100_bins", run_optimizer=False, weighted_prior=False)
    # samp.plot_mse_prior_likelihood()
    # samp = InferDens(z=8, num_bins=1000, nothing_off=False, mask_ionized=True, iter_num_max=10, plot_direc="data_adversarial_new_prior_1000_bins", run_optimizer=False, weighted_prior=False)
    # samp.plot_mse_prior_likelihood()
    # samp = InferDens(z=8, num_bins=256, nothing_off=False, mask_ionized=True, iter_num_max=10, plot_direc="data_adversarial_new_prior_256_bins", run_optimizer=False, weighted_prior=False)
    # samp.plot_mse_prior_likelihood()
    # for i in [229]:
    #     if i in [229, 509]:
    #         perc_ionized = 0.2
    #     else:
    #         perc_ionized = 0.1
    # plot_dir = "/Users/sabrinaberger/Current Research/just_grad_descent/z_7_perc_ionized_0.5_seed_1010_bins_128"
    # truth_field = np.load(plot_dir + "npy/truth_field_7.npy")
    # data = np.load(plot_dir + "npy/data_field_7.npy")
    samp = InferDens(seed=1010, z=7, truth_field=np.array([None]), brightness_temperature_field=np.array([None]),
                     num_bins=128, mask_ionized=False, iter_num_max=10, side_length=512,
                     plot_direc="", run_optimizer=False, weighted_prior=False, new_prior=True,
                     old_prior=False, nothing_off=False, verbose=False,
                     pspec_on_plot=True, debug=False, use_truth_mm=True, noise_off=True)

    # samp.plot_pspec_and_panel(normalize=True)
    samp.plot_panel(normalize=False)
    # samp.plot_pspecs()