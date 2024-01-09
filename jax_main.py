"""
Infrastucture to perform efficient, parallelized minimization of a likelihood based on
the Battaglia et al. 2013 paper model to infer density fields from brightness temperatures
Created October, 2022
Written by Sabrina Berger
independent_only_jax function written by Adrian Liu and Adélie Gorce
"""

# import jax related packages
import jax
import os
import scipy.interpolate as interpolate
import skimage
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
from jax.config import config
from jax.experimental.host_callback import id_print  # this is a way to print in Jax when things are preself.compiledd
from jax.scipy.optimize import \
    minimize as minimize_jax  # this is the bread and butter algorithm of this work, minimization using Jax optimized gradients
# from jax.example_libraries import optimizers as jaxopt
import jaxopt
import theory_matter_ps
from theory_matter_ps import circular_spec_normal, spherical_p_spec_normal

config.update("jax_enable_x64", True)  # this enables higher precision than default for Jax values
# config.update('jax_disable_jit', True) # this turns off jit compiling which is helpful for debugging

import matplotlib.pyplot as plt
import powerbox as pbox
import numpy as np
from jax_battaglia_full import Dens2bBatt
import matplotlib
from matplotlib.colors import SymLogNorm

# nice publication fonts for plots are implemented when these lines are uncommented
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})

### These flags could help when running on a GPU, but they didn't do anything for me.
# XLA_PYTHON_CLIENT_PREALLOCATE=false
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

### quick test below for install working of tanh gradient
# grad_tanh = grad(jnp.tanh)
# print(grad_tanh(110.))

class SwitchMinimizer:
    def __init__(self, seed, z, num_bins, likelihood_off=False, prior_off=False, data=None, truth_field=None, s_field=None, fixed_field=None,
                 noise_off=True, side_length=256, dimensions=2, debug=False, physical_side_length=256,
                 verbose=False, plot_direc="2D_plots", mask_ionized=False, weighted_prior=False, new_prior=False, old_prior=False, use_truth_mm=False,
                 create_instance=False):
        """
        TODO update
        :param z - the redshift you would like to create your density field at
        :param data (Default: None) - data that you're fitting your field to and will be used in your chi-squared.
                                If None, the data will be generated from the field meaning you know your truth field
                                beforehand.
        :param s_field (Default: None) - initial field to start density field optimization at.
                                If None, a Gaussian normal field will be used with sigma = 0.2 *
                                the standard deviation of the data.
        :param fixed_field (Default: None) - only used if include_param = True to keep the field constant while chi squared is
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
        # checking to make sure one of these three is true
        # assert np.count_nonzero(check_setup_bool_arr) == 1

        self.seed = seed
        self.z = z
        self.likelihood_off = likelihood_off
        self.prior_off = prior_off
        self.data = data
        self.truth_field = truth_field
        self.s_field = s_field
        self.fixed_field = fixed_field
        self.noise_off = noise_off
        self.side_length = side_length
        self.physical_side_length = physical_side_length
        self.dim = dimensions
        self.debug = debug
        self.verbose = verbose
        self.plot_direc = plot_direc
        self.iter_num = 0
        self.prior_count = 0
        self.likelihood_count = 0
        self.mask_ionized = mask_ionized
        self.num_bins = num_bins
        self.weighted_prior = weighted_prior
        self.new_prior = new_prior
        self.original_prior = old_prior
        self.use_truth_mm = use_truth_mm
        self.create_instance = create_instance

        ## new stuff
        self.resolution = self.side_length / self.physical_side_length # number of pixels / physical side length

        self.area = self.side_length**2
        # kmax = 2 * jnp.pi / self.physical_side_length * (self.side_length / 2)
        kmax = 50
        self.pspec_true = theory_matter_ps.get_truth_matter_pspec(kmax, self.physical_side_length, self.z, self.dim)
        ### get a tuple with the dimensions of the field
        self.size = []
        for i in range(self.dim):
            self.size.append(self.side_length)
        self.size = tuple(self.size)
        self.total_pixels = self.side_length ** self.dim  # total number of pixels
        ###############################################################################################################
        if self.seed == None:
            print("Seed is none.")
            return
        ### Debugging setup ###########################################################################################
        if self.debug:
            ### debugging arrays and saving iteration number
            self.likelihood_all = np.zeros((1000))
            self.value_all = np.zeros((self.side_length ** self.dim, 1000))
            self.prior_param_all = np.zeros((1000))
            self.likelihood_indiv_all = np.zeros((self.side_length ** self.dim, 1000))
            self.param_value_all = np.zeros((1000))
        ###############################################################################################################
        if self.create_instance:
            # just getting methods and instance
            return
        ### get *latent space* (z or unbiased field) and *data* only if self.data = np.array([None]) ##############################
        if self.data.any() == None and self.truth_field.any() == None:
            ### 1) create latent space (density field)
            pb_data_unbiased_field = self.create_better_normal_field(seed=self.seed)
            # truth field is just unbiased version made with pbox
            self.truth_field = pb_data_unbiased_field.delta_x()
            if self.original_prior: #old version
                self.fft_truth = self.fft_jax(self.truth_field)
                self.kvals_truth, self.pspec_2d_true = theory_matter_ps.convert_pspec_2_2D(self.pspec_true, self.side_length, self.z)
                # counts, pspec, kvals = self.p_spec_normal(self.pspec_2d_true, self.num_bins)
                # mask_kvals = kvals < 1.2
                # counts_func = interpolate.interp1d(kvals, counts, fill_value="extrapolate", bounds_error=False)
                # ensuring there are no 0s in the weights
                # mask_0 = self.kvals_truth == 0
                # self.kvals_truth[mask_0] = 0.01
                # self.weights = jnp.sqrt(counts_func(self.kvals_truth))
                self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(self.pspec_2d_true + 1j * self.pspec_2d_true)
                # self.weights_re, self.weights_im = self.INDICES_independent_only_jax(self.weights) # get im, re of weights
                # self.pspec_indep_nums_re *= self.weights_re
                # self.pspec_indep_nums_im *= self.weights_im

            elif self.new_prior:
                if self.dim == 2:
                    counts, self.pspec_box, k_vals_all = circular_spec_normal(self.truth_field, self.num_bins, self.resolution, self.area)
                if self.dim == 3:
                    counts, self.pspec_box, k_vals_all = spherical_p_spec_normal(self.truth_field, self.num_bins, self.resolution, self.area)

                plt.close()
                plt.scatter(k_vals_all, counts)
                plt.xlabel("k vals")
                plt.ylabel("counts")
                plt.xscale("log")
                plt.show()
                plt.close()
                self.weights = np.full_like(counts, 0.2)
                self.weights[k_vals_all > 12] = 1
                self.truth_final_pspec = self.pspec_true(k_vals_all)
                plt.close()
                plt.loglog(k_vals_all, self.truth_final_pspec, label="cmb generated")
                plt.loglog(k_vals_all, self.pspec_box, label="pspec of truth")
                plt.legend()
                plt.show()
                plt.close()


            else:
                print("ERROR")
                exit()
            ### 2) create data
            self.data = self.bias_field(self.truth_field)
            print("self.data")
            print(np.max(self.data))
        else: # data included in initialization of class
            # print("Using previously generated data and truth field.")
            assert(jnp.shape(self.truth_field)[0] != 0)
            assert(jnp.shape(self.data)[0] != 0)
            print("Not yet implemented")
            exit()
        ###############################################################################################################
        self.ionized_indices = jnp.argwhere(self.data.flatten() == 0).flatten()
        self.neutral_indices = jnp.argwhere(self.data.flatten() != 0).flatten()

        # generate diagonal matrices for chi-squared and adding noise if selected #####################################
        self.rms = 0.1
        if not self.noise_off:
            print("Added noise... NOTE THAT THIS COULD CAUSE DATA VALUES TO FALL BELOW 0.")
            # Assume that the noise is 10% of the rms of PRE-noised field, SCALE_NUM IS MULTIPLIED BY RMS IN FUNCTION
            self.data = self.data + self.create_jax_normal_field(100) # 100 is the seed of the noise (same each time)
        # self.rms_Tb = jnp.std(self.data)
        self.N_diag = self.rms ** 2 * jnp.ones((self.side_length ** self.dim))
        # print("N_diagonal")
        # id_print(self.N_diag)
        ###############################################################################################################

        # Generate a starting point field that is just 0.2 * rms of the data if it's not already specified
        if self.s_field == None:
            print("CREATING NEW STARTING FIELD.")
            # starting guess, using a powerbox field without flat pspec
            # self.s_field_original = self.create_normal_field(std_dev=np.std(self.truth_field))
            self.s_field_original = self.create_normal_field(std_dev=0.1)

            # self.pb = pbox.LogNormalPowerBox(
            #     N=self.side_length,  # number of wavenumbers
            #     dim=self.dim,  # dimension of box
            #     pk=lambda k: 0.1 * k**-2,  # The power-spectrum
            #     boxlength=self.physical_side_length,  # Size of the box (sets the units of k in pk)
            #     seed=seed  # Use the same seed as our powerbox
            #     #ensure_physical=True
            # )
            # self.s_field_original = self.pb.delta_x()

            # Currently feeding standard deviation of truth field, could change this to data / 27
            self.s_field = jnp.asarray(self.s_field_original.flatten())
            # print("USING TRUTH AS S FIELD")
            # self.s_field_original = self.truth_field
            # self.s_field = self.truth_field.flatten()

        ###############################################################################################################

    def create_normal_field(self, std_dev):
        """This method creates a numpy random field and converts it to a Jax array"""
        np_version = np.random.normal(scale=std_dev, size=self.size)
        return jnp.asarray(np_version)

    def create_jax_normal_field(self, seed=0):
        """This method creates a Jax random field with the default seed or the one specified."""
        seed = jax.random.PRNGKey(seed)
        return jax.random.normal(seed, shape=self.size)

    def rerun(self, likelihood_off, prior_off, mask_ionized):
        self.mask_ionized = mask_ionized
        self.likelihood_off = likelihood_off
        self.prior_off = prior_off
        # sets starting field as best old field
        if self.verbose:
            print("USING OLD STARTING FIELD.")
        self.s_field = jnp.copy(self.best_field_reshaped).flatten()
        if self.mask_ionized:
            self.preserve_original = jnp.copy(self.s_field)
        self.run_grad_descent()

    def run_grad_descent(self):
        # Start gradient descent ######################################################################################
        self.opt_result = self.differentiate_2D_func()
        self.best_field = self.opt_result.flatten()
        if self.mask_ionized:
            # put back the ionized regions which are the only things allowed to change
            assert jnp.shape(self.best_field) == (self.side_length**2,)
            self.preserve_original = self.preserve_original.at[self.ionized_indices].set(self.best_field[self.ionized_indices])
            self.best_field_reshaped = jnp.array(jnp.reshape(self.preserve_original, self.size))
        else:
            self.best_field_reshaped = jnp.array(jnp.reshape(self.best_field, self.size))
        ###############################################################################################################
        if self.debug:
            self.debug_likelihood()

    def differentiate_2D_func(self):
        ## field should already be flattened here
        func = self.chi_sq_jax
        if self.prior_off:
            self.likelihood_cand_fields = jnp.zeros((1000, self.side_length ** 2))
            self.iter_num_internal = 0
        ## ORIGINAL HERE
        # opt_result = minimize_jax(func, self.s_field, method='l-bfgs-experimental-do-not-rely-on-this', options={"maxiter": 1e100, "maxls": 1e100})
        # print("status number of run")
        # print(opt_result.status)
        # self.final_func_val = opt_result.fun
        # print("FINAL HESSIAN")
        # self.hessian_final = opt_result.hess_inv
        # print(opt_result.hess_inv)
        # print("number optimization")
        # print(opt_result.nit)
        # print("number funct")
        # print(opt_result.nfev)
        # print("jacobian")
        # print(opt_result.njev)
        # print("Was it successful?")
        # print(opt_result.success)
        # return opt_result.x
        #######

        # opt_result = jaxopt.ScipyMinimize(method="l-bfgs-b", fun=func, tol=1e-12)
        # bounds = jnp.repeat(np.asarray([-1, jnp.max(self.truth_field)]), self.side_length**2)
        # bounds = []
        # for i in range(self.side_length**2):
        #     bounds.append((-1, jnp.max(self.truth_field)))
        # self.mask_ionized = True
        # self.prior_off = False
        # self.likelihood_off = False
        # self.preserve_original = self.truth_field.flatten()
        # result = scipy.optimize.shgo(func, bounds=bounds, options={"maxfev": 10}) #, sampling_method='sobol') #minimizer_kwargs={"method": " L-BFGS-B"}

        ######## running jaxopt version
        # jaxopt.LBFGS.stop_if_linesearch_fails = True
        # , options = {"maxiter": 1e100, "maxls": 1e100})
        opt_result = jaxopt.LBFGS(fun=self.chi_sq_jax, tol=1e-12, maxiter=1000, maxls=1000, stop_if_linesearch_fails=True)
        params, state = opt_result.run(self.s_field)
        self.final_func_val = state.value

        if self.verbose:
            print("Was it successful?")
            print(opt_result.success)
            print("How many iterations did it take?")
            print(self.iter_num)

        return params

    def chi_sq_jax(self, guess):
        """
        This is the Seljak et al. chi^2, of the form
        s^t S^-1 s + [d - f(s, lambda)]^t N^-1 [d - f(s, lambda)]
        where s is the candidate field (that we are trying to fit for)
        S is the covariance matrix

        :param guess - Array of real numbers for s (the candidate field) and/or the parameter(s)
        """

        if self.mask_ionized:
            copy_guess = jnp.copy(guess.flatten())
            full_guess = jnp.copy(self.preserve_original.flatten())

            assert jnp.shape(copy_guess) == (self.side_length**2,)
            assert jnp.shape(full_guess) == (self.side_length**2,)

            full_guess = full_guess.at[self.ionized_indices].set(copy_guess[self.ionized_indices])
            candidate_field = jnp.reshape(full_guess, self.size)
        else:
            candidate_field = jnp.reshape(guess, self.size)

        # note param_init was passed in and is a constant
        discrepancy = self.data - self.bias_field(candidate_field)

        ## calculating intermediate pspec
        # fig_ps, ax_ps = plt.subplots()
        # kvals, pspec = helper_func.p_spec_normal()(self.truth_field, 60, self.side_length)
        # ax_ps.loglog(kvals, pspec, label="TRUTH", c="g")
        # kvals, pspec = helper_func.p_spec_normal()(candidate_field.primal, 60, self.side_length)
        # ax_ps.loglog(kvals, pspec, label=str(self.iter_num), c="k")
        # ax_ps.set_title("intermediate pspec - both prior + likelihood on")
        # ax_ps.set_ylim((1e-6, 1e2))
        # ax_ps.legend(loc=1)
        # fig_ps.savefig(f"png files/{self.iter_num}.png")
        # plt.close()
        ########

        #### get likelihood for all cases #############################################################################
        likelihood = jnp.dot(discrepancy.flatten() ** 2, 1. / self.N_diag)
        ###############################################################################################################

        if self.original_prior: # old version
            # FT and get only the independent modes
            self.fourier_box = self.fft_jax(candidate_field) * (1/self.weights**2)
            fourier_nums_real, fourier_nums_imag = self.independent_only_jax(self.fourier_box)

            real_prior = jnp.dot(fourier_nums_real**2,
                        2 / self.pspec_indep_nums_re)  # Half variance for real
            imag_prior = jnp.dot(fourier_nums_imag**2,
                        2 / self.pspec_indep_nums_im)  # Half variance for imag

            prior = real_prior + imag_prior

        elif self.new_prior:
            if self.dim == 2:
                counts, power_curr, k_values = circular_spec_normal(candidate_field, self.num_bins, self.resolution, self.area)
            elif self.dim == 3:
                counts, power_curr, k_values = spherical_p_spec_normal(candidate_field, self.num_bins, self.resolution, self.area)

            # sigma = counts**2
            x = (self.truth_final_pspec - power_curr).flatten()
            # prior = jnp.dot(x**2, 1/sigma)
            prior = np.sum(x**2)
            # sig = jnp.full_like(candidate_field, 0.1)
            # prior_gauss = jnp.sum(jnp.dot(candidate_field**2, 1/sig))
            # prior += prior_gauss
            # plt.close()
            # plt.title("individual candidate")
            # plt.plot(k_values, self.weights)
            # plt.show()
            # plt.close()


        if self.likelihood_off:
            # likelihood = 10**-2 *  likelihood
            likelihood = 0
        elif self.prior_off:
            prior = 0

        # if self.weighted_prior:
        #     self.likelihood = likelihood
        #     self.prior = prior
        #     likelihood = likelihood/jnp.log10(likelihood)
        #     prior = prior/jnp.log10(prior)
        #     self.final_likelihood_prior = likelihood + prior
        # else:
        #     mean_curr = jnp.abs(jnp.mean(candidate_field))
            # print("mean_curr")
            # id_print(mean_curr)


            # prior_extra = jnp.where(mean_curr < 1., 0, 1e5)

            # if mean_prior < 1.:
            #     prior_extra = 0
            # else:
            #     prior_extra = 1e5
            # self.likelihood = likelihood
            # self.prior = prior
        self.final_likelihood_prior = likelihood + prior

        if self.verbose:
            print("self.likelihood_off", self.likelihood_off)
            print("self.prior_off", self.prior_off)
            print("prior: ")
            id_print(prior)
            print("likelihood: ")
            id_print(likelihood)
            print("current final posterior")
            id_print(self.final_likelihood_prior)
        self.iter_num += 1
        return self.final_likelihood_prior

    def create_better_normal_field(self, seed):
        if self.use_truth_mm:
            print("Using truth matter power spectrum to generate field.")
            # def pspecify(k):
            #     k_shape = jnp.shape(k)
            #     if len(k_shape) < 1:
            #         return jnp.interp(jnp.asarray([k]), self.pspec_true_xp, self.pspec_true_fp)
            #     k = k.flatten()
            #     power_flat = jnp.interp(k, self.pspec_true_xp, self.pspec_true_fp)
            #     power_flat = jnp.reshape(power_flat, k_shape)
            #     return power_flat
            #
            # self.pspec_true = pspecify

            ## should only be used for setting an initial test field
            self.pb = pbox.LogNormalPowerBox(
                N=self.side_length,  # number of wavenumbers
                dim=self.dim,  # dimension of box
                pk=self.pspec_true,  # The power-spectrum
                boxlength=self.physical_side_length,  # Size of the box (sets the units of k in pk)
                seed=seed  # Use the same seed as our powerbox
                #ensure_physical=True
            )
        else:
            ## should only be used for setting an initial test field
            self.pb = pbox.LogNormalPowerBox(
                N=self.side_length,  # number of wavenumbers
                dim=self.dim,  # dimension of box
                pk=lambda k: 0.1 * k**-2,  # The power-spectrum
                boxlength=self.physical_side_length,  # Size of the box (sets the units of k in pk)
                seed=seed,  # Use the same seed as our powerbox
                # ensure_physical=True
            )
        return self.pb

    def ft_jax(self, x):
        return jnp.sum(jnp.abs(jnp.fft.fft(x)) ** 2)

    def bias_field(self, field):
        """
        Used to bias field or convert to temperature brightness.
        :param field -- field being converted
        :param param (Default: None) -- bias upon which to bias current field
        """
        batt_model_instance = Dens2bBatt(field, delta_pos=1, set_z=self.z, flow=True, resolution=self.resolution)
        # get neutral versus ionized count ############################################################################
        self.neutral_count = jnp.count_nonzero(batt_model_instance.X_HI)

        if self.debug:
            plt.close()
            plt.imshow(batt_model_instance.X_HI)
            plt.title("X_HI")
            plt.colorbar()
            plt.savefig(f"{self.plot_direc}/plots/{self.iter_num}_X_HI.png")
        if self.verbose:
            print("The number of neutral pixels is: " )
            id_print(self.neutral_count)

        ###############################################################################################################
        return batt_model_instance.temp_brightness

    def fft_jax(self, field):
        """ FFTs a field with proper FFT shifting
        :param field - field to be FFTed"""
        return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(field)))

    def independent_only_jax(self, box):
        """
        This takes a box that is meant to be a Fourier space representation
        of a box that is purely real in configuration space and outputs
        only the independent elements. It does so separately for the real
        and imaginary parts because there are more independent real parts
        than there are imaginary parts.

        THIS ONLY WORKS IN 2D RIGHT NOW
        """
        assert(len(jnp.shape(box)) > 1)
        N = box.shape[0]
        # First get the upper half plane minus the bits that aren't independent
        first_row = box[0, :-(N // 2 - 1)]
        sandwiched_rows = box[1:N // 2].flatten()
        origin_row = box[N // 2, :-(N // 2 - 1)]

        real_part = jnp.concatenate((jnp.real(first_row),
                                    jnp.real(sandwiched_rows),
                                    jnp.real(origin_row)))

        imag_part = jnp.concatenate((jnp.imag(first_row[1:-1]),
                                    jnp.imag(sandwiched_rows),
                                    jnp.imag(origin_row[1:-1])))

        return real_part, imag_part

    def INDICES_independent_only_jax(self, box):
        """
        This takes an array and gets the independent parts by index without taking the real or imaginary parts of them
        """
        assert(len(jnp.shape(box)) > 1)
        N = box.shape[0]
        # First get the upper half plane minus the bits that aren't independent
        first_row = box[0, :-(N // 2 - 1)]
        sandwiched_rows = box[1:N // 2].flatten()
        origin_row = box[N // 2, :-(N // 2 - 1)]

        real_part = jnp.concatenate((first_row,
                                    sandwiched_rows,
                                    origin_row))

        imag_part = jnp.concatenate((first_row[1:-1],
                                    sandwiched_rows,
                                    origin_row[1:-1]))

        return real_part, imag_part

    def debug_likelihood(self):
        """"Makes same plots to show likelihood and prior as a function of iteration"""
        fig, ax = plt.subplots(1, 2)
        plt.title(f"bias = {self.actual_bias}, side length = {self.side_length}")
        ax[0].plot(self.param_value_all[:self.iter_num], label="param")
        ax[0].hlines(y=self.actual_bias, xmin=0, xmax=self.iter_num, label="truth param", color='k')
        ax[0].set_xlabel("num iterations")
        ax[0].set_ylabel("value")
        ax[0].legend()

        ax[1].plot(self.likelihood_all[:self.iter_num], label="likelihood")
        ax[1].plot(self.prior_param_all[:self.iter_num], label="prior")
        ax[1].set_xlabel("num iterations")
        ax[1].set_yscale("log")
        ax[1].legend()
        plt.savefig(f"{self.plot_direc}/plots/param.png")
        plt.close()

        if self.dim == 2:
            value_all_reshaped = np.reshape(self.value_all[:, :self.iter_num],
                                            (self.side_length, self.side_length, self.iter_num))
            likelihood_all_reshaped = np.reshape(self.likelihood_indiv_all[:, :self.iter_num],
                                                 (self.side_length, self.side_length, self.iter_num))
            for i in range(self.side_length):
                for j in range(self.side_length):
                    fig, ax = plt.subplots(1, 2)
                    # ax[0].set_yscale("log")
                    ax[0].hlines(y=self.truth_field[i, j], color='k', xmin=0, xmax=self.iter_num)
                    ax[0].plot(value_all_reshaped[i, j, :self.iter_num])
                    ax[0].set_xlabel("num iterations")
                    ax[0].set_ylabel("value")
                    ax[1].set_yscale("log")
                    ax[1].plot(likelihood_all_reshaped[i, j, :self.iter_num], label="indiv chi squared")
                    ax[1].plot(self.likelihood_all[:self.iter_num], label="total chi squared")
                    ax[1].plot(self.prior_param_all[:self.iter_num], label="total prior")
                    ax[1].legend()
                    ax[1].set_xlabel("num iterations")

                    fig.suptitle(f"bias = {self.actual_bias}, side length = {self.side_length}")
                    plt.savefig(f"{self.plot_direc}/plots/pixel_num_{i * j}_check.png")
                    plt.close()

        else:
            for i in range(self.side_length ** self.dim):
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(self.value_all[i, :self.iter_num])
                ax[0].set_xlabel("num iterations")
                ax[0].set_ylabel("value")
                ax[1].set_yscale("log")
                ax[1].plot(self.likelihood_indiv_all[i, :self.iter_num], label="indiv chi squared")
                ax[1].plot(self.likelihood_all[:self.iter_num], label="total chi squared")
                ax[1].plot(self.prior_param_all[:self.iter_num], label="total prior")
                ax[1].legend()
                ax[1].set_xlabel("num iterations")
                fig.suptitle(f"bias = {self.actual_bias}, side length = {self.side_length}")
                plt.savefig(f"{self.plot_direc}/plots/pixel_num_{i}_check.png")
                plt.close()

