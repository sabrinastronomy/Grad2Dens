"""
Infrastructure to generate MAPs of the matter density field
Created November, 2022
Written by Sabrina Berger
"""

from jax_main import GradDescent
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy

# plotting pspec for each iteration
from jax_battaglia_full import Dens2bBatt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

### defaults for paper plots
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 9})


class InferDens(GradDescent):
    """
    This class allows you to pass in a brightness temperature field measurement at a particular redshift
    and infer the density field.
    """
    def __init__(self, z, num_bins=None, mask_ionized=False, nothing_off=False, brightness_temperature_field=None, plot_direc="data_adversarial_new_prior", side_length=256,
                 dimensions=2, iter_num_max=10, rest_num_max=3, noise_off=True, save_posterior_values=False, run_optimizer=False,
                 pspec_on_plot=False, mse_plot_on=False, weighted_prior=None):
        self.plot_direc = plot_direc # this overwrites the plot_directory in the GradDescent class
        self.z = z
        print(num_bins)
        super().__init__(self.z, num_bins=num_bins, data=brightness_temperature_field, side_length=side_length,
                               dimensions=dimensions, include_field=True, plot_direc=self.plot_direc,
                               indep_prior=True, verbose=True, noise_off=noise_off, weighted_prior=weighted_prior)
        self.iter_num_max = iter_num_max # number of iterations over which to minimize posterior
        self.rest_num_max = rest_num_max # number of iterations to rest on the given configuration
        self.save_posterior_values = save_posterior_values
        self.mse_vals = jnp.empty(self.iter_num_max)
        self.posterior_vals = jnp.empty(self.iter_num_max)
        self.pspec_on = pspec_on_plot
        self.mse_plot_on = mse_plot_on
        self.mask_ionized_after_first = mask_ionized
        self.mask_ionized = False
        self.nothing_off = nothing_off
        if run_optimizer:
            self.infer_density_field()

    def check_threshold(self):
        batt_model_instance = Dens2bBatt(self.best_field_reshaped, delta_pos=1, set_z=self.z, flow=True)
        temp_model_brightness = batt_model_instance.temp_brightness
        root_mse_curr = self.root_mse(self.data, temp_model_brightness)
        # if root_mse_curr > 1 and root_mse_curr < 150:
        #     print("Converged to less than 150 MSE.")
        #     # exit()
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
            if iter_num_big == 0:
                if self.nothing_off:
                    self.prior_off = False
                    self.likelihood_off = False
                else:
                    # first iteration, we keep just the likelihood on
                    self.prior_off = True
                    self.likelihood_off = False
                self.run_grad_descent()

                ### checking whether truth field returns higher posterior than best guessed field
                # print("TRUTH FIELD---------------------------------------")
                # self.chi_sq_jax(self.truth_field)
                #
                # print("best FIELD---------------------------------------")
                # self.chi_sq_jax(self.best_field_reshaped)
                #
                # print("starting FIELD---------------------------------------")
                # self.chi_sq_jax(self.s_field_original)

                self.check_field(self.truth_field, "truth_field", show=True, save=True,
                                 iteration_number=-1)

                self.check_field(self.data, "data", show=True, save=True,
                                 iteration_number=-1)

                self.check_field(self.best_field_reshaped, "data", show=True, save=True,
                                 iteration_number=-1)
                np.save(self.plot_direc + f"/npy/truth_field_{self.z}.npy", self.truth_field)
                np.save(self.plot_direc + f"/npy/data_field_{self.z}.npy", self.data)

            else:
                if not self.nothing_off: # if both on this is skipped
                    if rest_num == self.rest_num_max or self.prior_off:
                        self.prior_off = not self.prior_off
                        self.likelihood_off = not self.likelihood_off
                        rest_num = 0
                    self.mask_ionized = self.mask_ionized_after_first
                    print(self.mask_ionized)
                    # if self.z < 10:
                    #     self.mask_ionized = True
                    # else:
                    #     self.mask_ionized = False

                self.rerun(likelihood_off=self.likelihood_off, prior_off=self.prior_off, mask_ionized=self.mask_ionized)

            rest_num += 1
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

            # checking only at the end of a round of likelihood/prior off
            self.mse_vals = self.mse_vals.at[iter_num_big].set(self.check_threshold())

            if self.save_posterior_values:
                self.posterior_vals = self.posterior_vals.at[iter_num_big].set(self.final_likelihood_prior.primal)
                np.save(self.plot_direc + f"/npy/posterior_vals_{self.z}.npy",self.posterior_vals)

            np.save(self.plot_direc + f"/npy/mse_vals_{self.z}.npy", self.mse_vals)
            np.save(self.plot_direc + f"/npy/{self.plot_title}_best_field_{self.z}_{iter_num_big}.npy", self.best_field_reshaped)

            print("current iteration")
            print(iter_num_big)

    def plot_pspec_and_panel(self, tick_font_size=7, normalize=False):
        figsize = (8, 12)
        rows = 4
        cols = 2

        if normalize:
            truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
        else:
            truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
            data = np.load(self.plot_direc + f"/npy/data_field_{self.z}.npy")
            data_fig, data_axes = plt.subplots(rows, cols, figsize=figsize)

        if self.mse_plot_on:
            mse_vals = np.load(self.plot_direc + f"/npy/mse_vals_{self.z}.npy")

        if self.pspec_on:
            _, pspec_truth, kvals = self.p_spec_normal(truth_field, self.num_bins, self.side_length)
            fig_pspec, axes_pspec = plt.subplots()
            axes_pspec.loglog(kvals, pspec_truth, label=f"Truth field", lw=3, c="k")

        fig, axes = plt.subplots(rows, cols, figsize=figsize)


        # prior_off = True
        # likelihood_off = False
        k = 0
        rest_num = 0

        # make data/density field panels and pspec simultaneously
        for i in range(rows):
            for j in range(cols):
                print("plotting", (i,j))
                ### annoying labels
                if k == 0:
                    if self.nothing_off:
                        self.prior_off = False
                        self.likelihood_off = False
                    else:
                        # first iteration, we keep just the likelihood on
                        self.prior_off = True
                        self.likelihood_off = False

                else:
                    if not self.nothing_off:  # if both on this is skipped
                        if rest_num == self.rest_num_max or self.prior_off:
                            self.prior_off = not self.prior_off
                            self.likelihood_off = not self.likelihood_off
                            rest_num = 0
                        self.mask_ionized = self.mask_ionized_after_first
                rest_num += 1
                if not self.likelihood_off and not self.prior_off:
                    self.plot_title = "both_on"
                elif self.likelihood_off:
                    self.plot_title = "likelihood_off"
                elif self.prior_off:
                    self.plot_title = "prior_off"
                else:
                    print("Something is wrong as neither the likelihood or prior is on.")
                    exit()
                if i == 0 and j == 0:
                    title = "(not masked)\nlikelihood only "
                    if normalize:
                        hist, bin_edges = np.histogram(truth_field, bins=1000)
                        axes[i][j].plot(bin_edges[1:], hist)
                        axes[i][j].set_xscale('symlog')
                        axes[i][j].set_xlabel("truth field density")
                        axes[i][j].set_xlim(-10**1, 10**2)
                        continue
                    else:
                        im = axes[i][j].imshow(truth_field, vmin=-1, vmax=2)
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

                best_field = np.load(self.plot_direc + f"/npy/{self.plot_title}_best_field_{self.z}_{k}.npy")
                if normalize:
                    hist, bin_edges = np.histogram(best_field, bins=1000)
                    axes[i][j].plot(bin_edges[1:], hist)
                    axes[i][j].set_xscale('symlog')
                    axes[i][j].set_xlabel("density")
                    axes[i][j].set_xlim(-10 **1, 10 ** 2)

                else:
                    im = axes[i][j].imshow(best_field, vmin=-1, vmax=2)
                    divider = make_axes_locatable(axes[i][j])
                    cax = divider.append_axes('bottom', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, orientation="horizontal")
                    cbar.ax.tick_params(labelsize=tick_font_size)
                    t = axes[i][j].text(20, 220, f"Iteration #{k} \nwith " + title, color="black", weight='bold')
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
                    t = data_axes[i][j].text(20, 220, f"Iteration #{k} \nwith " + title, color="black", weight='bold')
                    t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                    data_axes[i][j].set_xticks([])
                    data_axes[i][j].set_aspect('equal')

                if self.pspec_on:
                    if k in [0, 2, 5, 10]:
                        counts, pspec_normal, kvals = self.p_spec_normal(best_field, self.num_bins, self.side_length)
                        counts, pspec_pre, kvals = self.p_spec_pre(best_field, self.num_bins, self.side_length)
                        # print(counts)
                        poisson_rms = np.sqrt(np.abs(pspec_normal - pspec_pre)) / np.sqrt(counts)
                        print(poisson_rms)
                        # axes_pspec.plot(kvals, pspec_normal)
                        axes_pspec.errorbar(kvals, pspec_normal, yerr=poisson_rms, label=f"Iteration # {k} with {title}", ls="--", alpha=0.3)
                        axes_pspec.set_xscale("log")
                        axes_pspec.set_yscale("log")

                k += 1

            if self.pspec_on:
                axes_pspec.legend()
                axes_pspec.set_ylabel("Power [units]")
                axes_pspec.set_xlabel("k [units]")
                fig_pspec.savefig(self.plot_direc + "/plots/pspec.png", dpi=300)
                plt.close(fig_pspec)

            # fig.tight_layout()

            if normalize:
                fig.savefig(self.plot_direc + "/plots/normalize_field_panel_iterations.png", dpi=300)
            else:
                fig.savefig(self.plot_direc + "/plots/field_panel_iterations.png", dpi=300)
                data_fig.tight_layout()
                data_fig.savefig(self.plot_direc + "/plots/data_panel_iterations.png", dpi=300)

                plt.close(data_fig)
            plt.close(fig)

        if self.mse_plot_on:
            ### plotting mse vals
            fig_mse, ax_mse = plt.subplots()
            length_of_runs = len(mse_vals[mse_vals > 0]) - 1

            ax_mse.semilogy(mse_vals[:length_of_runs], label="mean square error", ls=None)
            low_lim_ax, upper_lim_ax = ax_mse.get_ylim()[0], ax_mse.get_ylim()[1]

            ax_mse.vlines(3, low_lim_ax, upper_lim_ax, label="Iteration #3", color="blue")
            ax_mse.vlines(6, low_lim_ax, upper_lim_ax, label="Iteration #6", color="orange")
            ax_mse.vlines(9, low_lim_ax, upper_lim_ax, label="Iteration #9", color="green")
            ax_mse.vlines(12, low_lim_ax, upper_lim_ax, label="Iteration #12", color="red")

            ax_mse.set_xlim(None, length_of_runs)
            ax_mse.set_xlabel("iteration")
            ax_mse.legend()
            fig_mse.savefig(self.plot_direc + f"/plots/all_mse_post_{self.z}.png")

    def get_prior_likelihood_post(self):
        self.like_arr = np.zeros(self.iter_num_max)
        self.prior_arr = np.zeros(self.iter_num_max)

        truth_field = np.load(self.plot_direc + f"/npy/truth_field_{self.z}.npy")
        data = np.load(self.plot_direc + f"/npy/data_field_{self.z}.npy")
        counts, pspec_box, _ = self.p_spec_normal(truth_field, self.num_bins, self.side_length)
        rest_num = 0
        labels = []
        for iter_num_big in range(self.iter_num_max):
            ### annoying labels
            if iter_num_big == 0:
                if self.nothing_off:
                    self.prior_off = False
                    self.likelihood_off = False
                else:
                    # first iteration, we keep just the likelihood on
                    self.prior_off = True
                    self.likelihood_off = False

            else:
                if not self.nothing_off:  # if both on this is skipped
                    if rest_num == self.rest_num_max or self.prior_off:
                        self.prior_off = not self.prior_off
                        self.likelihood_off = not self.likelihood_off
                        rest_num = 0
                    self.mask_ionized = self.mask_ionized_after_first
            rest_num += 1
            if not self.likelihood_off and not self.prior_off:
                self.plot_title = "both_on"
            elif self.likelihood_off:
                self.plot_title = "likelihood_off"
            elif self.prior_off:
                self.plot_title = "prior_off"
            else:
                print("Something is wrong as neither the likelihood or prior is on.")
                exit()

            best_field = np.load(self.plot_direc + f"/npy/{self.plot_title}_best_field_{self.z}_{iter_num_big}.npy")


            candidate_field = jnp.reshape(best_field, self.size)

            # note param_init was passed in and is a constant
            discrepancy = data - self.bias_field(candidate_field)
            likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
            self.like_arr[iter_num_big] = likelihood
            #### prior
            counts, power_curr, _ = self.p_spec_normal(candidate_field, self.num_bins, self.side_length)
            sigma = counts ** 2
            x = (pspec_box - power_curr).flatten()
            prior = jnp.dot(x ** 2, 1 / sigma)
            self.prior_arr[iter_num_big] = prior
            labels.append(self.plot_title)

        print("final prior array")
        print(self.prior_arr)

        self.labels = labels
        np.save(self.plot_direc + f"/npy/prior_arr_{self.z}.npy", self.prior_arr)
        np.save(self.plot_direc + f"/npy/like_arr_{self.z}.npy", self.like_arr)



    def plot_mse_prior_likelihood(self):
        ### plotting mse vals
        mse_vals = np.load(self.plot_direc + f"/npy/mse_vals_{self.z}.npy")
        self.get_prior_likelihood_post()
        self.prior_arr = np.load(self.plot_direc + f"/npy/prior_arr_{self.z}.npy")
        self.like_arr = np.load(self.plot_direc + f"/npy/like_arr_{self.z}.npy")
        plt.close()
        plt.semilogy(self.prior_arr, label="prior")
        plt.semilogy(self.like_arr, label="likelihood")
        plt.semilogy(mse_vals, label="mse")
        plt.xlim(0, 10)
        plt.legend()
        plt.show()


        fig_mse, ax_mse = plt.subplots()

        ax_mse.semilogy(mse_vals, label="MSE new prior", ls=None)
        ax_mse.semilogy(self.prior_arr, label="prior", ls=None)
        ax_mse.semilogy(self.like_arr, label="likelihood", ls=None)


        # ax_mse.set_xlim(None, 350)
        low_lim_ax, upper_lim_ax = ax_mse.get_ylim()[0], ax_mse.get_ylim()[1]

        for iter_num_big in range(self.iter_num_max):
            ax_mse.vlines(iter_num_big, low_lim_ax, upper_lim_ax, ls="--", color="k")
            ax_mse.text(iter_num_big - 0.4, low_lim_ax, self.labels[iter_num_big], fontsize=12, rotation="vertical", alpha=0.5)


        ax_mse.set_xlabel("iteration")
        ax_mse.legend()
        fig_mse.savefig(self.plot_direc + f"/plots/prior_likelihood_mse_post_{self.z}.png")


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

    samp = InferDens(z=8, num_bins=100, nothing_off=False, mask_ionized=True, iter_num_max=20,
                     plot_direc="data_adversarial_new_prior_100_bins_no_weighted_prior", run_optimizer=True,
                     weighted_prior=False)
    # samp.plot_pspec_and_panel(normalize=True)
    # samp.plot_pspec_and_panel(normalize=False)

    samp.plot_mse_prior_likelihood()

