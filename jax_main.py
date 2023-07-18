"""
Infrastucture to perform efficient, parallelized gradient descent of both included toy models and
the Battaglia et al. 2013 paper model to generate temperature brightness fields from density fields.
Created October, 2022
Written by Sabrina Berger and Adrian Liu
"""

# import jax related packages
import jax
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
from jax.config import config
from jax.experimental.host_callback import id_print  # this is a way to print in Jax when things are precompiled
from jax.scipy.optimize import \
    minimize as minimize_jax  # this is the bread and butter algorithm of this work, minimization using Jax optimized gradients

config.update("jax_enable_x64", True)  # this enables higher precision than default for Jax values
config.update('jax_disable_jit', True)

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import powerbox as pbox
import numpy as np
from jax_battaglia_full import Dens2bBatt
import scipy.stats as stats
import helper_func

# nice publication fonts for plots are implemented when these lines are uncommented
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


### These flags could help when running on a GPU, but they didn't do anything for me.
# XLA_PYTHON_CLIENT_PREALLOCATE=false
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'

### quick test below for install working of tanh gradient
# grad_tanh = grad(jnp.tanh)
# print(grad_tanh(110.))

class GradDescent:
    def __init__(self, z, battaglia, likelihood_off=False, prior_off=False, data=None, truth_field=None, s_field=None, fixed_field=None, noise_off=True,
                 side_length=256, dimensions=2, indep_prior=False, include_param_and_field=False, include_field=False,
                 include_param=False, debug=False, toy_model_dict={"actual_bias": None, "param_init": None,
                                                                   "dependence": False, "step_function": False},
                 compile=False, vectorize=False, verbose=False, plot_direc="2D_plots", internal_alternating=False,
                 easy=False, mask_ionized=False):
        """
        :param z - the redshift you would like to create your density field at
        :param battaglia - Boolean that determines whether or not a toy model or Battaglia et al. model is on
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
        :param include_param_and_field (Default: False) - reduce chi squared with both parameter and field. This doesn't work except in simple cases.
        :param include_field (Default: False): reduce chi squared with field while parameter is held fixed
        :param include_param (Default: False) - reduce chi squared with only the parameter while the field is held fixed
        :param debug (Default: False) - turns off Jit, saves all intermediate likelihood and prior values
        :param toy_model_dict (Default: {"actual_bias": None, "param_init": None, "dependence": False, "step_function": False}) - get relevant information for the toy model, toy models only work in 2D
        :param compile (Default: False) - whether or not to use a jitted version for the minimization
        :param vectorize (Default: False) - whether or not to use vectorized version for the minimization (Not working)
        :param verbose (Default: False) - whether or not to print a bunch of stuff
        :param plot_direc (Default: 2D_plots) - where to save plots
        """
        # checking to make sure one of these three is true
        check_setup_bool_arr = [include_field, include_param, include_param_and_field]
        assert np.count_nonzero(check_setup_bool_arr) == 1

        self.z = z
        self.battaglia = battaglia
        self.likelihood_off = likelihood_off
        self.prior_off = prior_off
        self.data = data
        self.truth_field = truth_field
        self.s_field = s_field
        self.fixed_field = fixed_field
        self.noise_off = noise_off
        self.side_length = side_length
        self.dim = dimensions
        self.indep_prior = indep_prior
        self.include_param_and_field = include_param_and_field
        self.include_field = include_field
        self.include_param = include_param
        self.debug = debug
        self.toy_model_dict = toy_model_dict
        self.compile = compile
        self.verbose = verbose
        self.plot_direc = plot_direc
        self.iter_num = 0
        self.prior_count = 0
        self.likelihood_count = 0
        self.internal_alternating = internal_alternating # whether or not to turn the prior on and off inside the grad descent
        self.easy = easy
        self.mask_ionized = mask_ionized
        ### get a tuple with the dimensions of the field
        self.size = []
        for i in range(self.dim):
            self.size.append(self.side_length)
        self.size = tuple(self.size)
        self.total_pixels = self.side_length ** self.dim  # total number of pixels

        ### Toy model setup ###########################################################################################
        if not self.battaglia:
            self.toy_model_dict = toy_model_dict
            self.actual_bias = self.toy_model_dict["actual_bias"]
            self.param_init = self.toy_model_dict[
                "param_init"]  # either fixed parameter or initial parameter to start guessing from in toy model
            self.dependence = self.toy_model_dict["dependence"]
            self.step_function = self.toy_model_dict["step_function"]
            if self.dependence:
                self.field_type = "smoothed"
            elif self.step_function:
                self.field_type = "step_fun"
            else:
                self.field_type = "constant"
        else:
            self.param_init = None
        ###############################################################################################################

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
        if self.data == None:
            ### 1) create latent space
            pb_data_unbiased_field = self.create_better_normal_field(bias_k_dependence=False)
            # truth field is just unbiased version made with pbox
            self.truth_field = pb_data_unbiased_field.delta_x()
            # We also need the power spectrum box only for the independent prior To do, should this be biased or unbiased
            self.fft_truth = self.fft_jax(self.truth_field)
            self.pspec_box = jnp.abs(self.fft_truth)**2
            self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(self.pspec_box + 1j * self.pspec_box)


            ### 2) create data
            if not self.battaglia:  # Toy model data comes from pbox!
                print("Entered incorrect else")
                pb_data_biased = self.create_better_normal_field(bias_k_dependence=True)
                self.truth_field = jnp.asarray(pb_data_biased.delta_x())
            else:  # Battaglia et al. data comes from Dens2bBatt class
                self.data = self.bias_field(self.truth_field, None)
        else: # data included in initialization of class
            assert(jnp.shape(self.truth_field)[0] != 0)
            assert(jnp.shape(self.data)[0] != 0)
            self.fft_truth = self.fft_jax(self.truth_field)
            self.pspec_box = jnp.abs(self.fft_truth)**2
            self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(self.pspec_box + 1j * self.pspec_box)

        ###############################################################################################################

        # generate diagonal matrices for chi-squared and adding noise if selected #####################################
        self.rms = 5
        if not self.noise_off:
            print("Added noise...")
            # Assume that the noise is 10% of the rms of PRE-noised field, SCALE_NUM IS MULTIPLIED BY RMS IN FUNCTION
            self.data = self.data + self.create_normal_field(scale_num=0.1)
        self.N_diag = self.rms ** 2 * jnp.ones((self.side_length ** self.dim))
        ###############################################################################################################

        # Generate a starting point field that is just 0.2 * standard deviation of the data if it's not already specified
        if self.s_field == None and not self.easy:
            print("CREATING NEW STARTING FIELD.")
            if self.battaglia:  # TODO figure this out
                self.rms = 1
            # starting guess, using a powerbox field without flat pspec
            self.s_field_original = self.create_normal_field(scale_num=0.1)
            # self.s_field_original = self.create_better_normal_field(bias_k_dependence=False, seed=2000, bias=0.1).delta_x()  # starting guess, using a powerbox field without flat pspec
            # self.s_field_original = self.data / jnp.max(self.data) + 1
            self.s_field = jnp.asarray(self.s_field_original.flatten())
            self.check_field(self.s_field_original, "starting field", show=True)

        else:
            print("USING OLD STARTING FIELD.")
            self.s_field = jnp.asarray(self.s_field).flatten()
            if self.mask_ionized:
                self.preserve_original = jnp.copy(self.s_field)
        ###############################################################################################################

        # Start gradient descent ######################################################################################
        if compile and not vectorize:  ## these two don't work for all options
            ("Running jitted version.")
            self.jit_vec_minimize = jax.jit(self.differentiate_2D_func)
            self.opt_result = self.jit_vec_minimize()
            self.result = jnp.array(jnp.reshape(self.opt_result, self.size))
        elif compile and vectorize:  ## these two don't work for all options
            print("Running vectorized and jitted version.")
            self.jit_vec_minimize = jax.jit(jax.vmap(self.differentiate_2D_func))
            data_tiled = jnp.tile(self.data, [256, 256])
            self.opt_result = self.jit_vec_minimize(data_tiled)
            self.result = jnp.array(jnp.reshape(self.opt_result[0], self.size))
        else:
            print("Running non-vectorized and non-jitted version.")
            self.opt_result = self.differentiate_2D_func()
            if self.include_param_and_field:  # both field and parameter are not fixed
                self.best_param = self.opt_result.at[-1].get()
                self.best_field = self.opt_result.at[:-1].get()
                self.best_field_reshaped = jnp.array(jnp.reshape(self.best_field, self.size))
            elif self.include_param:  # field fixed
                self.best_param = self.opt_result.at[0].get()
                self.best_field = self.fixed_field  # just the fixed field we passed in initially
            elif self.include_field:  # parameter fixed
                self.best_param = self.param_init  # just the fixed parameter we passed in initially
                self.best_field = self.opt_result
                if self.mask_ionized:
                    # put back the ionized regions
                    self.preserve_original = self.preserve_original.at[self.ionized_indices].set(self.best_field)
                    self.best_field_reshaped = jnp.array(jnp.reshape(self.preserve_original, self.size))
                else:
                    self.best_field_reshaped = jnp.array(jnp.reshape(self.opt_result, self.size))
            else:
                print("Something went really wrong here. How did you get past the assertions?")
        ###############################################################################################################
        if self.debug:
            self.debug_likelihood()
        if self.verbose:
            print("Final chi squared value: ")
            id_print(self.final_likelihood_prior)

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
            fig.savefig(f"{self.plot_direc}/3_panel_z_{self.z}_no_noise_.png", dpi=300)
        else:
            fig.savefig(f"{self.plot_direc}/3_panel_z_{self.z}_w_noise.png", dpi=300)
        plt.close()

    def create_normal_field(self, scale_num):
        """This method creates a numpy random field and converts it to a Jax array"""
        np_version = np.random.normal(scale=scale_num * self.rms, size=self.size)
        return jnp.asarray(np_version)

    def create_jax_normal_field(self, seed=0):
        """This method creates a Jax random field with the default seed or the one specified."""
        seed = jax.random.PRNGKey(seed)
        return jax.random.normal(seed, shape=self.size)

    def create_better_normal_field(self, bias_k_dependence, seed=1010, bias=0.1):
        ## should only be used for setting an initial test field
        if bias_k_dependence:
            if self.battaglia:
                print("You shouldn't be getting your data from power box. Something's wrong.")
            self.pb = pbox.PowerBox(
                N=self.side_length,  # number of wavenumbers
                dim=self.dim,  # dimension of box
                pk=lambda k: self.bias_k(k ** 2, self.actual_bias) ** 2 * k ** -2.,
                boxlength=1.,  # Size of the box (sets the units of k in pk)
                a=0,  # a and b need to be set like this to properly match numpy's fft
                b=2 * jnp.pi,
                seed=seed,
                ensure_physical=True)  # Set a seed to ensure the box looks the same every time (optional)
        else:
            if self.battaglia:
                print("generating pbox")
                self.pb = pbox.LogNormalPowerBox(
                    N=self.side_length,  # number of wavenumbers
                    dim=self.dim,  # dimension of box
                    pk=lambda k: bias * k ** -2.,  # The power-spectrum
                    boxlength=128.0,  # Size of the box (sets the units of k in pk)
                    seed=seed,  # Use the same seed as our powerbox
                    # a = 0,  # a and b need to be set like this to properly match numpy's fft
                    # b = 2 * jnp.pi
                )
            else:
                self.pb = pbox.PowerBox(
                    N=self.side_length,  # number of wavenumbers
                    dim=self.dim,  # dimension of box
                    pk=lambda k: k ** -2.,
                    boxlength=1.,  # Size of the box (sets the units of k in pk)
                    a=0,  # a and b need to be set like this to properly match numpy's fft
                    b=2 * jnp.pi,
                    seed=1010,
                    ensure_physical=True)  # Set a seed to ensure the box looks the same every time (optional)
        return self.pb

    def ft_jax(self, x):
        return jnp.sum(jnp.abs(jnp.fft.fft(x)) ** 2)

    def bias_k(self, k_modes_squared, param):
        """
        Creating a bias function dependent on k
        :param k_modes_squared - the frequencies (k-modes) returned by the fft, i.e., np.fftfreq.
        """
        if self.dependence:
            bias = jnp.exp(-0.5 * k_modes_squared * param ** 2)
            return bias
        elif self.step_function:
            # 0 for large k-modes
            # 1 for high k-modes
            bias = jnp.where(k_modes_squared > param, 0, 1)  # issue with NonConcreteBooleans in JAX, need this syntax
            return bias
        else:
            return param

    def bias_field(self, field, param=None):
        """
        Used to bias field or convert to temperature brightness.
        :param field -- field being converted
        :param param (Default: None) -- bias upon which to bias current field
        """
        if self.battaglia:
            batt_model_instance = Dens2bBatt(field, delta_pos=1, set_z=self.z, flow=True)
            # get neutral versus ionized count ############################################################################
            self.neutral_count = np.count_nonzero(batt_model_instance.X_HI)

            if self.debug:
                plt.close()
                plt.imshow(batt_model_instance.X_HI)
                plt.title("X_HI")
                plt.colorbar()
                plt.savefig(str({self.iter_num}) + "_X_HI.png")
            print("The number of neutral pixels is: " + str(self.neutral_count) + ".")
            print(
                f"The box is {100 * self.neutral_count / self.total_pixels} % neutral")  # incorrect unless you have X_HI
            ###############################################################################################################
            return batt_model_instance.temp_brightness
        else:
            print("Entered incorrect else.")
            #### below is toy model stuff
            if self.include_param_and_field:
                print("I entered the wrong if statement, and I'm cheating!")
                # cheat way -- already knows parameter
                param = self.actual_bias
            if self.dependence or self.step_function:
                # Smoothed or muted k modes in a field for biasing. Note this only works in 2D.
                if field.ndim == 2:
                    ffted_field = jnp.fft.fftn(field)
                    fft_k_1 = jnp.fft.fftfreq(self.side_length, d=1 / self.side_length)
                    k1, k2 = jnp.meshgrid(fft_k_1, fft_k_1)  # need 2D k grid
                    fft_k = k1 ** 2 + k2 ** 2  # should be square root I think
                    biased_k = self.bias_k(fft_k, param)
                    new_field = jnp.fft.ifftn(ffted_field * biased_k)
                    return jnp.real(new_field)
                elif field.ndim == 1:
                    ffted_field = jnp.fft.fftn(field)
                    size = jnp.shape(field)
                    fft_k = jnp.fft.fftfreq(size[0],
                                            d=1 / self.side_length)  # d is sampling time, need 1/sampling time for sampling frequency
                    biased_k = self.bias_k(fft_k ** 2, param)
                    new_field = jnp.fft.ifftn(ffted_field * biased_k)
                    return jnp.real(new_field)
            else:
                # constant/easy bias
                return field * param

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
        plt.savefig(f"{self.plot_direc}/param.png")
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
                    plt.savefig(f"{self.plot_direc}/pixel_num_{i * j}_check.png")
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
                plt.savefig(f"plots/pixel_num_{i}_check.png")
                plt.close()

    def chi_sq_jax(self, guess):
        """
        This is the Seljak et al. chi^2, of the form
        s^t S^-1 s + [d - f(s, lambda)]^t N^-1 [d - f(s, lambda)]
        where s is the candidate field (that we are trying to fit for)
        S is the covariance matrix

        :param guess - Array of real numbers for s (the candidate field) and/or the parameter(s)
        """

        # Convert the fields back to arrays with proper dimensionality
        if self.include_param_and_field:  # both param and field vary
            param = guess.at[-1].get()
            candidate_field = guess.at[:-1].get()
            candidate_field = jnp.reshape(candidate_field, self.size)
            biased_field_curr = self.bias_field(candidate_field, param)
            discrepancy = self.data - biased_field_curr
        elif self.include_param:  # field fixed
            param = guess.at[0].get()
            biased_field_curr = self.bias_field(self.fixed_field, param)
            biased_field_curr_reshaped = jnp.reshape(biased_field_curr,
                                                     self.size)  # note fixed_field was passed in and is a constant
            discrepancy = self.data - biased_field_curr_reshaped
        elif self.include_field:  # param fixed
            if self.mask_ionized:
                copy_guess = jnp.copy(guess)
                full_guess = jnp.copy(self.preserve_original)
                full_guess = full_guess.at[self.ionized_indices].set(copy_guess)
                candidate_field = jnp.reshape(full_guess, self.size)
            else:
                candidate_field = jnp.reshape(guess, self.size)

            if self.battaglia:
                # note param_init was passed in and is a constant
                discrepancy = self.data - self.bias_field(candidate_field)
            else:
                discrepancy = self.data - self.bias_field(candidate_field, self.param_init)

        ## calculating intermediate pspec
        # fig_ps, ax_ps = plt.subplots()
        # kvals, pspec = helper_func.bin_density_new(self.truth_field, 60, self.side_length)
        # ax_ps.loglog(kvals, pspec, label="TRUTH", c="g")
        # kvals, pspec = helper_func.bin_density_new(candidate_field.primal, 60, self.side_length)
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

        if self.indep_prior:
            check_neg = jnp.argwhere(candidate_field < -1).flatten()
            print(check_neg)
            if True:
                # FT and get only the independent modes
                fourier_box = self.fft_jax(candidate_field)
                fourier_nums_real, fourier_nums_imag = self.independent_only_jax(fourier_box)
                # Do the same for the power spectrum box. Assigning an imaginary part to the
                # original power spectrum box is just a trick to store pspec in the right format
                # to be compared to the imaginary parts
                # fourier_nums_real = fourier_nums_real.at[:-1].get()
                # pspec_indep_nums_re = pspec_indep_nums_re.at[:-1].get()
                # Work out the prior term
                # if self.mask_ionized:
                #     sort_arg = np.argsort(fourier_nums_real)
                #     fourier_nums_real = fourier_nums_real[sort_arg]
                #     self.pspec_indep_nums_re = self.pspec_indep_nums_re[sort_arg]
                #
                #     fourier_nums_imag = fourier_nums_imag[sort_arg]
                #     self.pspec_indep_nums_im = self.pspec_indep_nums_im[sort_arg]
                #
                #     real_prior = jnp.dot(fourier_nums_real[-100:] ** 2, (2 / self.pspec_indep_nums_re[-100:]))  # Half variance for real
                #     imag_prior = jnp.dot(fourier_nums_imag[-100:] ** 2, (2 / self.pspec_indep_nums_im[-100:]))  # Half variance for imag
                # else:
                real_prior = jnp.dot(fourier_nums_real ** 2, (2 / self.pspec_indep_nums_re))  # Half variance for real
                imag_prior = jnp.dot(fourier_nums_imag ** 2, (2 / self.pspec_indep_nums_im))  # Half variance for imag
                    # print(real_prior)
                    # print(imag_prior)
            # if False: # new as of 1/12/23
            #     # exp(- (P - P_true) ^ 2 / P_true ^ 2)
            #     curr_field_fft = self.fft_jax(candidate_field)
            #     curr_pspec_box = curr_field_fft ** 2
            #     curr_pspec_indep_nums_re, curr_pspec_indep_nums_im = self.independent_only_jax(curr_pspec_box + 1j * curr_pspec_box)
            #     id_print(curr_pspec_indep_nums_re-self.pspec_indep_nums_re)
            #     id_print(curr_pspec_indep_nums_im-self.pspec_indep_nums_im)
            #
            #     real_prior = jnp.sum((curr_pspec_indep_nums_re - self.pspec_indep_nums_re) ** 2)
            #     imag_prior = jnp.sum((curr_pspec_indep_nums_im - self.pspec_indep_nums_im) ** 2)
            #     # real_prior = jnp.dot((curr_pspec_indep_nums_re-self.pspec_indep_nums_re) ** 2, (1 / self.pspec_indep_nums_re))  # Half variance for real
            #     # imag_prior = jnp.dot((curr_pspec_indep_nums_im-self.pspec_indep_nums_im) ** 2, (1 / self.pspec_indep_nums_im))  # Half variance for real

            prior = real_prior + imag_prior
        else:
            if self.include_param:  # fixed field
                sigma_param = 0.1
                prior_param = (0.5 / sigma_param ** 2) * ((param - 3.2) ** 2)
                prefactor_param = np.log(1 / (np.sqrt(2 * np.pi) * sigma_param))
                prior = prior_param - prefactor_param
            elif self.include_param_and_field:
                sigma_param = 0.1
                prior_param = (0.5 / sigma_param ** 2) * ((param - 3) ** 2)
                prefactor_param = np.log(1 / (np.sqrt(2 * np.pi) * sigma_param))
                prior_param_only = prior_param - prefactor_param
                sigma_D = 10
                prior_rhos = (0.5 / sigma_D ** 2) * jnp.sum((candidate_field.flatten() ** 2))
                prefactor_prior = -(self.side_length ** self.dim) / 2 * jnp.log(2 * jnp.pi * sigma_D)
                field_prior = prior_rhos - prefactor_prior
                prior = prior_param_only + field_prior
            elif self.include_field:  # fixed param
                # use
                # independent
                # modes

                k_arr, p_arr = self.bin_density(self.truth_field, bin=False)
                # self.bin_density(candidate_field, bin=True)

                if self.iter_num == 0:
                    self.cov = self.create_cov(k_arr)[1:]

                    id_print(self.cov)
                p_arr = p_arr[1:]
                cov_inv = jnp.diag(1 / self.cov)

                first_prior = jnp.matmul(jnp.transpose(jnp.conjugate(p_arr)), cov_inv)
                prior = jnp.matmul(first_prior, p_arr)
                prior = jnp.abs(prior)

        if self.debug:
            likelihood_all = discrepancy.flatten() ** 2 * 1. / self.N_diag
            self.likelihood_all[self.iter_num] = likelihood.primal
            self.prior_param_all[self.iter_num] = prior.primal
            self.likelihood_indiv_all[:, self.iter_num] = likelihood_all.primal
            flattened = candidate_field.flatten()
            self.value_all[:, self.iter_num] = flattened.primal
            self.param_value_all[self.iter_num] = param.primal

        if self.internal_alternating:
            if self.prior_count > 47:
                self.likelihood_off = False
                self.prior_off = True
                self.prior_count = 0
            if self.likelihood_count > 6:
                self.likelihood_off = True
                self.prior_off = False
                self.likelihood_count = 0

            if self.likelihood_off:
                self.prior_count += 1
            if self.prior_off:
                self.likelihood_count += 1
        else:
            if self.likelihood_off:
                likelihood = 0
            elif self.prior_off:
                prior = 0



        self.final_likelihood_prior = likelihood + prior

        if self.verbose:
            print("self.likelihood_off", self.likelihood_off)
            print("self.prior_off", self.prior_off)
            print("self.prior_count", self.prior_count)
            print("self.likelihood_count", self.likelihood_count)
            print("prior: ")
            id_print(prior)
            print("likelihood: ")
            id_print(likelihood)
            print("current final posterior")
            id_print(self.final_likelihood_prior)
        self.iter_num += 1
        # if len(check_neg) > 0:
        #     self.final_likelihood_prior += jnp.inf
        return self.final_likelihood_prior

    def differentiate_2D_func(self):
        ## field should already be flattened here
        func = self.chi_sq_jax
        print("Attempting gradient descent now.")
        if self.include_param_and_field:  # field and param
            param_init = self.param_init
            x0 = jnp.append(self.s_field, param_init)
        elif self.include_param:
            param_init = float(self.param_init)
            x0 = jnp.asarray([param_init, 0])
        elif self.include_field:  # just field
            if self.easy: #### WARNING THIS IS CHEATING
                print("YOU ARE MAKING THE GRADIENT DESCENT VERY EASY.")
                self.s_field_original = self.truth_field + np.random.rand(*np.shape(self.truth_field))
                x0 = jnp.asarray(self.s_field_original.flatten())
                self.check_field(self.s_field_original, "starting field", show=True)

            elif self.mask_ionized:
                self.ionized_indices = jnp.argwhere(self.data.flatten() == 0).flatten()
                id_print(self.ionized_indices)
                x0 = self.s_field[self.ionized_indices]
            else:
                x0 = self.s_field
        opt_result = minimize_jax(func, x0=x0, method='l-bfgs-experimental-do-not-rely-on-this')
        print("Was it successful?")
        print(opt_result.success)
        print("How many iterations did it take?")
        print(self.iter_num)
        return opt_result.x

    def grad_2D_func(self):
        func = self.chi_sq_jax
        opt_result = jax.grad(func)
        return opt_result(self.s_field)

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
        if iteration_number >= 0:
            plt.title(f"(Iteration #{iteration_number}) z = " + str(self.z) + " " + title)
        else:
            plt.title("z = " + str(self.z) + " " + title)

        if "field" in title and self.dim == 1:
            id_print(self.data)
            inds = jnp.where(self.data == 0)[0]
            print(inds)
            for ind in inds:
                plt.scatter(ind, field[ind], color="black")
            plt.ylim(-1.5, 15)
            plt.legend()

        if save:
            plt.savefig(f"{self.plot_direc}/{title}_{self.z}" + "_battaglia.png")
        if show:
            plt.show()

        plt.close()

    def power(self):
        input = self.truth_field * 1  # scaling amplitude, Mpc
        fft_data = jnp.fft.fftn(input)
        power_data = jnp.abs(fft_data) ** 2  # Mpc^2
        k_arr = jnp.fft.fftfreq(self.side_length, 1) * 2 * np.pi  # extra 2pi needed here
        k1, k2 = jnp.meshgrid(k_arr, k_arr)  # need 2D k grid
        fft_k = jnp.sqrt(k1 ** 2 + k2 ** 2)  # should be square root I think

        power_data = power_data.flatten()
        fft_k = fft_k.flatten()

        kbins = np.arange(0, self.side_length // 2 + 1, 1.)
        Abins, _, _ = stats.binned_statistic(fft_k, power_data,
                                             statistic="mean",
                                             bins=kbins)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        self.p_spec = jnp.concatenate((Abins[::-1], Abins))
        print(np.shape(self.p_spec))
        plt.plot(kvals.primal, Abins.primal)
        plt.title("power spectrum")
        plt.show()

    def create_cov(self, dens_k_flat):
        k_vals, power = self.bin_density(self.truth_field, bin=True)
        power = jnp.asarray(power)
        k_vals = jnp.asarray(k_vals)
        plt.loglog(k_vals, power, label="truth field")
        # plt.show()
        # scipy.interpolate.interp1d(k_vals. power)
        # id_print(dens_k_flat)
        dens_k_flat = dens_k_flat.at[0].set(1e-5)
        power = lambda k: 0.1 * k ** -2.
        # id_print(dens_k_flat[0])
        # id_print(power(dens_k_flat[0]))
        # def get_ind_lambda(k):
        #     ind = jnp.abs(k_vals-k).argmin() # Todo interpolate?
        #     return power.at[ind].get()
        # diag_cov = jax.vmap(get_ind_lambda)(dens_k_flat)
        plt.loglog(dens_k_flat, power(dens_k_flat), label="compute")
        plt.legend()
        plt.title("power cov")
        plt.show()
        return power(dens_k_flat)

    # @staticmethod
    # def binning_power(i, Abins, indices_actual_array, fft_data_squared):
    #     count = Abins[i]
    #     mask = indices_actual_array == i
    #     total_power_in_bin = jnp.sum(fft_data_squared[mask])
    #     return total_power_in_bin / count

    def bin_density(self, field, bin=True):

        if not bin:
            fft_data = jnp.fft.fftn(field)
            fft_data_squared = fft_data ** 2  # NOT ABSOLUTE JUST TAKING DELTA ^ 2
            k_arr = jnp.fft.fftfreq(self.side_length) * 2 * jnp.pi
            k1, k2 = jnp.meshgrid(k_arr, k_arr)
            k_mag_full = jnp.sqrt(k1 ** 2 + k2 ** 2)
            k_mag = k_mag_full.flatten()
            fft_data_squared = fft_data_squared.flatten()
            return k_mag, fft_data_squared
        else:
            p_k_field, bins_field = pbox.get_power(field, self.side_length)

            # plt.loglog(bins_field.flatten(), 0.1 * bins_field.flatten()**-2, label="actual field")
            # plt.loglog(bins_field.flatten(), p_k_field.flatten(), label="current field")
            #
            # plt.legend()
            # plt.show()
            return bins_field.flatten(), p_k_field.flatten()

        # fft_data = jnp.fft.fftn(field)
        # fft_data_squared = jnp.abs(fft_data) ** 2
        # k_arr = jnp.fft.fftfreq(self.side_length, sampling_frequency) * 128  # extra 2pi needed here
        # k1, k2 = jnp.meshgrid(k_arr, k_arr)  # need 2D k grid
        # fft_k = jnp.sqrt(k1 ** 2 + k2 ** 2)  # should be square root I think
        # fft_k = jnp.asarray(fft_k.flatten())
        # fft_data_squared = jnp.asarray(fft_data_squared.flatten())
        # if not bin:
        #     return fft_k, fft_data_squared
        #
        # Abins, bin_edges = jnp.histogram(fft_k, self.side_length // 2 + 1)
        # indices_actual_array = jnp.digitize(fft_k, bin_edges)
        # average_power = jnp.empty(jnp.shape(bin_edges))
        # time_start = time.time()
        # print("shape of abins")
        # id_print(jnp.shape(Abins))
        # id_print(Abins.at[240].get())
        # for i in range(self.side_length // 2 + 1):
        #     count = Abins[i]
        #     mask = indices_actual_array == i
        #     total_power_in_bin = jnp.sum(fft_data_squared[mask])
        #     average_power = average_power.at[i].set(total_power_in_bin/count)
        # print("for loop time: ", time.time() - time_start)
        #
        # kvals = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # gets middle of bin k value
        #
        # power = average_power[1:]
        # # power = average_power[1:] * jnp.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        # # power /= self.side_length ** self.dim
        # if self.iter_num == 0:
        #     plt.plot(kvals, power)
        #     plt.show()
        # else:
        #     print(self.iter_num)
        #     plt.plot(kvals.primal.get(), power.primal.get())
        #     plt.show()
        # print("kvals")
        # id_print(kvals)

# if __name__ == "__main__":
#     import time
#     nsides = [8]
#     dimensions = 2
#     times = []
#
#     for nside in nsides:
#         print(f"Trying box with the following attributes: \n dimensions = {dimensions} \n sidelength = {nside}")
#         bias = 3.2
#         start = time.time()
#         ## Todo put a nice test case here
#         GradDescent2D()
#         end = time.time()
#         total_time = end - start
#         times.append(total_time)
