"""
Infrastucture to perform efficient, parallelized gradient descent of both included toy models and
the Battaglia et al. 2013 paper model to generate temperature brightness fields from density fields.
Created October, 2022
Written by Sabrina Berger and Adrian Liu
"""

# import jax related packages
import jax
import os
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
from jax.config import config
from jax.experimental.host_callback import id_print  # this is a way to print in Jax when things are preself.compiledd
from jax.scipy.optimize import \
    minimize as minimize_jax  # this is the bread and butter algorithm of this work, minimization using Jax optimized gradients
# from jax.example_libraries import optimizers as jaxopt
import jaxopt

config.update("jax_enable_x64", True)  # this enables higher precision than default for Jax values
# config.update('jax_disable_jit', True) # this turns off jit compiling which is helpful for debugging

import matplotlib.pyplot as plt
import powerbox as pbox
import numpy as np
from jax_battaglia_full import Dens2bBatt
# nice publication fonts for plots are implemented when these lines are uncommented
import matplotlib

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

class GradDescent:
    def __init__(self, seed, z, num_bins, likelihood_off=False, prior_off=False, data=None, truth_field=None, s_field=None, fixed_field=None,
                 noise_off=True, side_length=256, dimensions=2, debug=False,
                 verbose=False, plot_direc="2D_plots", mask_ionized=False, weighted_prior=False, new_prior=False, old_prior=False):
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
        # self.start_field_seed = start_field_seed

        ### get a tuple with the dimensions of the field
        self.size = []
        for i in range(self.dim):
            self.size.append(self.side_length)
        self.size = tuple(self.size)
        self.total_pixels = self.side_length ** self.dim  # total number of pixels
        ###############################################################################################################
        if self.seed == None:
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

        ### get *latent space* (z or unbiased field) and *data* only if self.data = None ##############################
        if self.data.any() == None and self.truth_field.any() == None:
            ### 1) create latent space
            pb_data_unbiased_field = self.create_better_normal_field(seed=self.seed)
            # truth field is just unbiased version made with pbox
            self.truth_field = pb_data_unbiased_field.delta_x()
            # We also need the power spectrum box only for the independent prior To do, should this be biased or unbiased
            if self.original_prior: #old version
                self.fft_truth = self.fft_jax(self.truth_field)
                self.pspec_box = jnp.abs(self.fft_truth)**2
                self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(self.pspec_box + 1j * self.pspec_box)
            elif self.new_prior:
                counts, self.pspec_box, _ = self.p_spec_normal(self.truth_field, self.num_bins)
            else:
                print("ERROR")
                exit()
            ### 2) create data
            self.data = self.bias_field(self.truth_field)
        else: # data included in initialization of class
            assert(jnp.shape(self.truth_field)[0] != 0)
            assert(jnp.shape(self.data)[0] != 0)
            if self.original_prior:  #old version
                self.fft_truth = self.fft_jax(self.truth_field)
                self.pspec_box = jnp.abs(self.fft_truth)**2
                self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(self.pspec_box + 1j * self.pspec_box)
            elif self.new_prior:
                counts, self.pspec_box, _ = self.p_spec_normal(self.truth_field, self.num_bins)
            else:
                print("ERROR")
                exit()

        ###############################################################################################################
        self.ionized_indices = jnp.argwhere(self.data.flatten() == 0).flatten()
        self.neutral_indices = jnp.argwhere(self.data.flatten() != 0).flatten()

        # generate diagonal matrices for chi-squared and adding noise if selected #####################################
        self.rms = 5
        if not self.noise_off:
            print("Added noise...")
            # Assume that the noise is 10% of the rms of PRE-noised field, SCALE_NUM IS MULTIPLIED BY RMS IN FUNCTION
            self.data = self.data + self.create_jax_normal_field(100)
        self.N_diag = self.rms ** 2 * jnp.ones((self.side_length ** self.dim))
        ###############################################################################################################

        # Generate a starting point field that is just 0.2 * standard deviation of the data if it's not already specified
        if self.s_field == None:
            print("CREATING NEW STARTING FIELD.")
            self.rms = 1
            # starting guess, using a powerbox field without flat pspec
            self.s_field_original = self.create_normal_field(scale_num=0.1)
            # self.s_field_original = self.create_better_normal_field(self.start_field_seed).delta_x()
            self.s_field = jnp.asarray(self.s_field_original.flatten())
            # self.check_field(self.s_field_original, "starting field", show=False)

        ###############################################################################################################
        self.run_grad_descent()

    def create_normal_field(self, scale_num):
        """This method creates a numpy random field and converts it to a Jax array"""
        np_version = np.random.normal(scale=scale_num * self.rms, size=self.size)
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
        ### only works for first
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


    def p_spec_normal(self, field, nbins):
        """
        square before averaging (histogramming)
        """
        # print(jnp.shape(field))
        # print(self.side_length)
        assert jnp.shape(field) == (self.side_length, self.side_length)
        fft_data = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(field)))
        fft_data_squared = jnp.abs(fft_data) ** 2
        k_arr = jnp.fft.fftshift(jnp.fft.fftfreq(self.side_length)) * 2 * jnp.pi
        k1, k2 = jnp.meshgrid(k_arr, k_arr)
        k_mag_full = jnp.sqrt(k1 ** 2 + k2 ** 2)

        counts, bin_edges = jnp.histogram(k_mag_full, nbins)
        binned_power, _ = jnp.histogram(k_mag_full, nbins, weights=fft_data_squared)
        kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        pspec = binned_power / counts / (self.side_length ** 2)
        return counts, pspec, kvals

    def p_spec_pre(self, field, nbins):
        """
        square after averaging (histograming)
        """
        fft_data = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(field)))
        fft_data = jnp.abs(fft_data)
        k_arr = jnp.fft.fftshift(jnp.fft.fftfreq(self.side_length)) * 2 * jnp.pi
        k1, k2 = jnp.meshgrid(k_arr, k_arr)
        k_mag_full = jnp.sqrt(k1 ** 2 + k2 ** 2)

        counts, bin_edges = jnp.histogram(k_mag_full, nbins)
        binned_fft, _ = jnp.histogram(k_mag_full, nbins, weights=fft_data)
        kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        pspec = binned_fft**2 / counts / (self.side_length ** 2)
        return counts, pspec, kvals



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
        """5
        This is the Seljak et al. chi^2, of the form
        s^t S^-1 s + [d - f(s, lambda)]^t N^-1 [d - f(s, lambda)]
        where s is the candidate field (that we are trying to fit for)
        S is the covariance matrix

        :param guess - Array of real numbers for s (the candidate field) and/or the parameter(s)
        """
        if self.prior_off:
            copy_guess = jnp.copy(guess.flatten())
            # self.likelihood_cand_fields = self.likelihood_cand_fields.at[self.iter_num_internal].set(copy_guess.primal)
            self.iter_num_internal += 1
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
            fourier_box = self.fft_jax(candidate_field)
            fourier_nums_real, fourier_nums_imag = self.independent_only_jax(fourier_box)
            real_prior = jnp.dot(fourier_nums_real ** 2, (2 / self.pspec_indep_nums_re))  # Half variance for real
            imag_prior = jnp.dot(fourier_nums_imag ** 2, (2 / self.pspec_indep_nums_im))  # Half variance for imag
            prior = real_prior + imag_prior
        elif self.new_prior:
            counts, power_curr, _ = self.p_spec_normal(candidate_field, self.num_bins)
            sigma = counts**2
            x = (self.pspec_box - power_curr).flatten()
            prior = jnp.dot(x**2, 1/sigma)


        if self.debug:
            likelihood_all = discrepancy.flatten() ** 2 * 1. / self.N_diag
            self.likelihood_all[self.iter_num] = likelihood.primal
            self.prior_param_all[self.iter_num] = prior.primal
            self.likelihood_indiv_all[:, self.iter_num] = likelihood_all.primal
            flattened = candidate_field.flatten()
            self.value_all[:, self.iter_num] = flattened.primal
            self.param_value_all[self.iter_num] = param.primal

        ### trying adrian prior here #####
        # fft_truth = self.fft_jax(candidate_field)
        # pspec_box_cand = jnp.abs(fft_truth) ** 2
        # sigma = 1
        # x = (self.pspec_box - pspec_box_cand).flatten()
        # denom = jnp.full(jnp.shape(x), 2*sigma**2)
        #
        # prior = jnp.dot(x**2, 1/denom)
        ### trying adrian prior here #####
        # id_print(likelihood)
        # id_print(prior)

        if self.likelihood_off:
            print("prior")
            id_print(prior)
            likelihood = 0
        elif self.prior_off:
            print("likelihood")
            id_print(likelihood)

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


    def create_better_normal_field(self, seed, bias=0.1):
        ## should only be used for setting an initial test field
        self.pb = pbox.LogNormalPowerBox(
            N=self.side_length,  # number of wavenumbers
            dim=self.dim,  # dimension of box
            pk=lambda k: bias * k ** -2.,  # The power-spectrum
            boxlength=self.side_length,  # Size of the box (sets the units of k in pk)
            seed=seed,  # Use the same seed as our powerbox
            # a = 0,  # a and b need to be set like this to properly match numpy's fft
            # b = 2 * jnp.pi
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
        batt_model_instance = Dens2bBatt(field, delta_pos=1, set_z=self.z, flow=True)
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

    def check_field(self, field, title, normalize=False, save=True, show=False, iteration_number=-1):
        # plotting
        plt.close()
        if normalize:
            field = (field - jnp.min(field)) / (jnp.max(field) - jnp.min(field))
        if field.ndim < 2:
            plt.plot(field)
        else:
            plt.imshow(field)
            plt.colorbar()
            # plt.clim(-1, 3)
        if iteration_number >= 0:
            t = plt.text(20, 240, f"Iteration #{iteration_number} with " + title)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        else:
            t = plt.text(20, 240, "z = " + str(self.z) + " " + title)
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))

        plt.tight_layout()
        if save:
            if iteration_number >= 0:
                plt.clim(-1, jnp.max(self.truth_field))
                plt.savefig(f"{self.plot_direc}/plots/" + f"iter_num_{iteration_number}_" + f"{self.z}" + "_battaglia.png")
            else:
                plt.savefig(f"{self.plot_direc}/plots/" + f"{title}_{self.z}" + "_battaglia.png")

        if show:
            plt.show()
        plt.close()

    def plot_3_panel(self):
        """This method gives a nice three panel plot showing the data, predicted field, and truth field"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        im1 = ax1.imshow(self.data)
        ax1.set_title("Observed data")
        im2 = ax2.imshow(self.result)
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