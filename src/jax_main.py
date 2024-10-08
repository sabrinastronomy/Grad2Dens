"""
Infrastucture to perform efficient, parallelized minimization of a likelihood based on
the Battaglia et al. 2013 paper model to infer density fields from brightness temperatures
Created October, 2022
Written by Sabrina Berger
independent_only_jax function written by Adrian Liu and Adélie Gorce
"""

# import jax related packages
import jax
import jax.numpy as jnp  # use jnp for jax numpy, note that not all functionality/syntax is equivalent to normal numpy
from jax.config import config
from jax.experimental.host_callback import id_print, id_tap  # this is a way to print in Jax when things are preself.compiledd
from jax.scipy.optimize import \
    minimize as minimize_jax  # this is the bread and butter algorithm of this work, minimization using Jax optimized gradients
# from jax.example_libraries import optimizers as jaxopt
import jaxopt
import theory_matter_ps
from theory_matter_ps import circular_spec_normal, spherical_p_spec_normal, after_spherical_p_spec_normal

config.update("jax_enable_x64", True)  # this enables higher precision than default for Jax values
config.update('jax_disable_jit', False) # this turns off jit compiling which is helpful for debugging
config.update("jax_debug_nans", True)


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
    def __init__(self, config_params, s_field):

        # checking to make sure one of these three is true
        # assert np.count_nonzero(check_setup_bool_arr) == 1

        self.config_params = config_params
        self.s_field = s_field

        ## new stuff
        self.resolution = self.config_params.side_length / self.config_params.physical_side_length  # number of pixels / physical side length
        print("resolution ", self.resolution)
        self.area = self.config_params.side_length ** 2
        self.volume = self.config_params.side_length**3
        # kmax = 2 * jnp.pi / self.physical_side_length * (self.side_length / 2)
        kmax = 50
        if self.config_params.use_truth_mm:
            self.pspec_true = theory_matter_ps.get_truth_matter_pspec(kmax, self.config_params.physical_side_length, self.config_params.z, self.config_params.dim)
        ### get a tuple with the dimensions of the field
        self.size = []
        for i in range(self.config_params.dim):
            self.size.append(self.config_params.side_length)
        self.size = tuple(self.size)
        self.total_pixels = self.config_params.side_length ** self.config_params.dim  # total number of pixels
        ###############################################################################################################
        if self.config_params.seed == None:
            print("Seed is none.")
            return
        ### Debugging setup ###########################################################################################
        if self.config_params.debug:
            ### debugging arrays and saving iteration number
            self.likelihood_all = np.zeros((1000))
            self.value_all = np.zeros((self.config_params.side_length ** self.config_params.dim, 1000))
            self.prior_param_all = np.zeros((1000))
            self.likelihood_indiv_all = np.zeros((self.config_params.side_length ** self.config_params.dim, 1000))
            self.param_value_all = np.zeros((1000))
        ###############################################################################################################
        if self.config_params.create_instance:
            # just getting methods and instance
            return
        ### get *latent space* (z or unbiased field) and *data* only if self.data = np.array([None]) ##############################
        if self.config_params.data == None and self.config_params.truth_field == None:
            ### 1) create latent space (density field)
            pb_data_unbiased_field = self.create_better_normal_field(seed=self.config_params.seed).delta_x()
            # truth field is just unbiased version made with pbox
            self.truth_field = jnp.asarray(pb_data_unbiased_field)
            # self.truth_field = pb_data_unbiased_field
            # if self.old_prior: #old version
            # calculating both regardless of prior
            if self.config_params.use_truth_mm:  # use theory matter power spectrum in prior
                if self.config_params.old_prior:
                    self.kvals_truth, self.pspec_2d_true = theory_matter_ps.convert_pspec_2_2D(self.pspec_true,
                                                                                               self.config_params.side_length,
                                                                                               self.config_params.z)
                    self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(
                        self.pspec_2d_true + 1j * self.pspec_2d_true)



            else:  # use individual realization of a field's power spectrum in prior
                self.fft_truth = self.fft_jax(self.truth_field)
                self.pspec_box = jnp.abs(self.fft_truth) ** 2
                self.pspec_indep_nums_re, self.pspec_indep_nums_im = self.independent_only_jax(
                    self.pspec_box + 1j * self.pspec_box)

            # elif self.new_prior:
            if self.config_params.dim == 2:
                counts, self.p_spec_truth_realization, k_vals_all = circular_spec_normal(self.truth_field,
                                                                                         self.config_params.num_bins,
                                                                                         self.resolution,
                                                                                         self.area)
            if self.config_params.dim == 3:
                counts, self.p_spec_truth_realization, k_vals_all = spherical_p_spec_normal(self.truth_field, self.config_params.num_bins, self.resolution, self.volume)

            if self.config_params.use_truth_mm:
                self.truth_final_pspec = self.pspec_true(k_vals_all)


            plt.close()
            plt.loglog(k_vals_all, self.p_spec_truth_realization, label="everything")
            mask_high_only = k_vals_all > 2
            plt.loglog(k_vals_all[mask_high_only], self.p_spec_truth_realization[mask_high_only], label="upper ks only")
            plt.legend()
            plt.show()
            plt.close()

            plt.close()
            plt.loglog(k_vals_all, counts)
            plt.xlabel("k vals")
            plt.ylabel("counts")
            plt.show()
            plt.close()
            ### 2) create data
            self.data = self.bias_field(self.truth_field)



        else:  # data included in initialization of class
            # print("Using previously generated data and truth field.")
            assert (jnp.shape(self.config_params.truth_field)[0] != 0)
            assert (jnp.shape(self.config_params.data)[0] != 0)
            self.data = self.config_params.data
            self.truth_field = self.config_params.truth_field
            print("Not yet implemented")
            exit()
        ###############################################################################################################
        # print("Ionized pixels if < 15")
        self.ionized_indices = jnp.argwhere(self.data < 5)
        self.neutral_indices = jnp.argwhere(self.data > 5)

        self.ionized_indices_flattened = jnp.argwhere(self.data.flatten() < 5).flatten()
        self.neutral_indices_flattened = jnp.argwhere(self.data.flatten() > 5).flatten()

        self.ionized_indices_mask = (self.data < 5).flatten()
        self.neutral_indices_mask = (self.data > 5).flatten()
        # self.truth_field = self.truth_field.at[self.ionized_indices].set(0)

        # generate diagonal matrices for chi-squared and adding noise if selected #####################################
        self.rms = 0.1
        if not self.config_params.noise_off:
            print("Added noise... NOTE THAT THIS COULD CAUSE DATA VALUES TO FALL BELOW 0.")
            # Assume that the noise is 10% of the rms of PRE-noised field, SCALE_NUM IS MULTIPLIED BY RMS IN FUNCTION
            self.data = self.data + self.create_jax_normal_field(100)  # 100 is the seed of the noise (same each time)
        # self.rms_Tb = jnp.std(self.data)
        self.N_diag = self.rms ** 2 * jnp.ones((self.config_params.side_length ** self.config_params.dim))

        print("N_diagonal")
        id_print(self.N_diag)
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

    def run(self, likelihood_off, prior_off, mask_ionized, use_old_field, iter_num_big):

        self.mask_ionized = mask_ionized
        self.likelihood_off = likelihood_off
        self.prior_off = prior_off
        self.iter_num_big = iter_num_big

        print("iter num")
        print(self.iter_num_big)
        # sets starting field as best old field
        if self.config_params.verbose:
            print("USING OLD STARTING FIELD.")
        if use_old_field:
            self.s_field = jnp.copy(self.best_field_reshaped).flatten()

        if self.mask_ionized:
            self.preserve_original = jnp.copy(self.s_field)

        # print("trying with assumption that we know neutral pixels")
        # truth_field_flattened = self.truth_field.flatten()
        # self.s_field = self.s_field.at[self.neutral_indices_flattened].set(truth_field_flattened[self.neutral_indices_flattened])

        # # print("-----------------")
        self.run_grad_descent()

    def run_grad_descent(self):
        # Start gradient descent ######################################################################################
        self.opt_result = self.differentiate_2D_func()
        self.best_field = self.opt_result.flatten()

        if self.mask_ionized:
            # put back the ionized regions which are the only things allowed to change
            assert jnp.shape(self.best_field) == (self.config_params.side_length ** self.config_params.dim,)
            self.preserve_original = self.preserve_original.at[self.ionized_indices_flattened].set(
                self.best_field[self.ionized_indices_flattened])

            self.best_field_reshaped = jnp.array(jnp.reshape(self.preserve_original, self.size))
        else:
            self.best_field_reshaped = jnp.array(jnp.reshape(self.best_field, self.size))
        ###############################################################################################################
        if self.config_params.debug:
            self.debug_likelihood()

    def differentiate_2D_func(self):
        ## field should already be flattened here
        self.iter_num = 0
        ######## running jaxopt version
        # jaxopt.LBFGS.stop_if_linesearch_fails = True
        # , options = {"maxiter": 1e100, "maxls": 1e100})
        self.likelihoods = jnp.empty(1000)
        self.priors = jnp.empty(1000)
        opt_result = jaxopt.LBFGS(fun=self.chi_sq_jax, tol=1e-12, maxiter=1000, maxls=1000,
                                  stop_if_linesearch_fails=True)


        params, state = opt_result.run(self.s_field)
        # print('state options')
        # print(state)
        self.final_func_val = state.value

        print("How many iterations did it take?")
        print(self.iter_num)

        plt.close()
        plt.plot(self.likelihoods, ls=None, label="likelihoods")
        plt.plot(self.priors, ls=None, label="priors")
        plt.legend()
        plt.savefig(self.plot_direc + f"/plots/iter_num_{self.iter_num_big}.png")
        plt.close()


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

            assert jnp.shape(copy_guess) == (self.config_params.side_length ** self.config_params.dim,)
            assert jnp.shape(full_guess) == (self.config_params.side_length ** self.config_params.dim,)


            full_guess = full_guess.at[self.ionized_indices_flattened].set(copy_guess[self.ionized_indices_flattened])
            candidate_field = jnp.reshape(full_guess, self.size)
        else:
            candidate_field = jnp.reshape(guess, self.size)


        # note param_init was passed in and is a constant
        discrepancy = self.data - self.bias_field(candidate_field)

        #### get likelihood for all cases #############################################################################
        x = (discrepancy.flatten() ** 2) * 1. / self.N_diag
        likelihood = jnp.mean(x)
        ###############################################################################################################
        # want circular/spherical pspec regardless of which prior we use
        if self.config_params.dim == 2:
            counts, power_curr, k_values = circular_spec_normal(candidate_field, self.config_params.num_bins, self.resolution,
                                                                self.area)
        elif self.config_params.dim == 3:
            counts, power_curr, k_values = spherical_p_spec_normal(candidate_field, self.config_params.num_bins, self.resolution,
                                                                   self.area)
        # also want difference between the candidate and truth field
        if self.config_params.use_truth_mm:
            x = (self.truth_final_pspec - power_curr).flatten()
        else:
            x = (self.p_spec_truth_realization - power_curr).flatten()

        if self.config_params.old_prior:  # old version
            # FT and get only the independent modes
            self.fourier_box = self.fft_jax(candidate_field)
            fourier_nums_real, fourier_nums_imag = self.independent_only_jax(self.fourier_box)

            real_prior = jnp.dot(fourier_nums_real ** 2,
                                 2 / self.pspec_indep_nums_re)  # Half variance for real
            imag_prior = jnp.dot(fourier_nums_imag ** 2,
                                 2 / self.pspec_indep_nums_im)  # Half variance for imag

            prior = (real_prior + imag_prior) / self.side_length**self.dim
        elif self.config_params.new_prior:
            if self.config_params.dim == 2:
                counts, power_curr, k_values = circular_spec_normal(candidate_field, self.config_params.num_bins, self.resolution, self.area)
            elif self.config_params.dim == 3:
                counts, power_curr, k_values = spherical_p_spec_normal(candidate_field,  self.config_params.num_bins, self.resolution, self.volume)
                after_counts, after_power_curr, after_k_values = after_spherical_p_spec_normal(candidate_field, self.config_params.num_bins, self.resolution, self.volume)
            x = (self.truth_final_pspec - power_curr).flatten()
            sigma = counts ** 2
            prior = jnp.mean((x**2) * sigma)
            print("prior multiplies sigma")
        
        if self.likelihood_off:
            # likelihood = 10**-2 *  likelihood
            likelihood = 0
        elif self.prior_off:
            prior = 0

        self.final_likelihood_prior = prior + likelihood

        if self.config_params.verbose:

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
        if self.config_params.use_truth_mm:
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
                N=self.config_params.side_length,  # number of wavenumbers
                dim=self.config_params.dim,  # dimension of box
                pk=self.pspec_true,  # The power-spectrum
                boxlength=self.config_params.physical_side_length,  # Size of the box (sets the units of k in pk)
                seed=seed  # Use the same seed as our powerbox
                # ensure_physical=True
            )
        else:
            ## should only be used for setting an initial test field
            self.pb = pbox.PowerBox(
                N=self.config_params.side_length,  # number of wavenumbers
                dim=self.config_params.dim,  # dimension of box
                pk=lambda k: 0.1 * k ** -3.5,  # The power-spectrum
                boxlength=self.config_params.physical_side_length,  # Size of the box (sets the units of k in pk)
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
        batt_model_instance = Dens2bBatt(field, delta_pos=1, set_z=self.config_params.z, flow=True, free_params=self.config_params.free_params)
        # get neutral versus ionized count ############################################################################
        self.neutral_count = jnp.count_nonzero(batt_model_instance.X_HI)

        if self.config_params.debug:
            plt.close()
            plt.imshow(batt_model_instance.X_HI)
            plt.title("X_HI")
            plt.colorbar()
            plt.savefig(f"{self.config_params.plot_direc}/plots/{self.config_params.iter_num}_X_HI.png")
        if self.config_params.verbose:
            print("The number of neutral pixels is: ")
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
        assert (len(jnp.shape(box)) > 1)
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
        assert (len(jnp.shape(box)) > 1)
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
