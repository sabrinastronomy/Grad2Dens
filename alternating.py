"""
Infrastructure to generate MAPs of the matter density field
Created November, 2022
Written by Sabrina Berger and Adrian Liu
"""

from jax_main import GradDescent as GD
import numpy as np
import matplotlib.pyplot as plt
# plotting pspec for each iteration
from helper_func import bin_density_new
side_length = 128
dim = 2
prev_likelihood = np.inf
param_going = True
z = 8

rest_num = 0
rest_num_max = 3
prior_off = True
likelihood_off = False
short = True
fig_ps = plt.figure()
frame1 = fig_ps.add_axes((.1, .3, .8, .6))

for iter_num in range(10):
    # print("------------------------------------------------------------------------------------------------------")
    if iter_num == 0:
        start = GD(z, battaglia=True, side_length=side_length, prior_off=prior_off, likelihood_off=likelihood_off,
                   dimensions=dim, include_field=True, plot_direc="adversarial",
                   indep_prior=True,
                   verbose=True, internal_alternating=False, easy=False)
        data = start.data
        s_field = start.best_field_reshaped
        truth_field = start.truth_field

        start.check_field(start.s_field_original, "initial field", show=True, save=True)
        start.check_field(start.data, "data", show=True, save=True)
        start.check_field(start.truth_field, "truth field", show=True, save=True)
        start.check_field(start.best_field_reshaped, "guessed field - likelihood on only", show=True, save=True, iteration_number=iter_num)
        prior_off = False
        likelihood_off = True
        kvals, pspec_best = bin_density_new(start.best_field_reshaped, 60, side_length)
        plt.loglog(kvals, pspec_best, label=f"initial guess", c="k")
    else:
        start = GD(z, data=data, s_field=s_field, prior_off=prior_off, likelihood_off=likelihood_off,
                   truth_field=truth_field,
                   battaglia=True, side_length=side_length, dimensions=dim, include_field=True,
                   plot_direc="adversarial",
                   indep_prior=True, verbose=True, mask_ionized=short)
        s_field = start.best_field_reshaped
        if likelihood_off and not prior_off:
            start.check_field(start.best_field_reshaped, "prior on - guessed field", show=True, save=True, iteration_number=iter_num)
        elif prior_off and not likelihood_off:
            start.check_field(start.best_field_reshaped, "likelihood on - guessed field", show=True, save=True, iteration_number=iter_num)
        else:
            start.check_field(start.best_field_reshaped, "both on - guessed field", show=True, save=True, iteration_number=iter_num)


        # np.save(f"{start.plot_direc}/guessed_field_z_{z}_{side_length}_{iter_num}_lp_{likelihood_off}_{prior_off}.npy", start.best_field_reshaped)
        if rest_num == rest_num_max or prior_off:
            # if rest_num == rest_num_max:
            prior_off = not prior_off
            likelihood_off = not likelihood_off
            rest_num = 0
            short = True
        rest_num += 1




        kvals, pspec_best = bin_density_new(start.best_field_reshaped, 60, side_length)
        plt.loglog(kvals, pspec_best, label=f"iteration #{iter_num}", c="k", alpha=0.1)



        np.save(f"best_field_{z}_{iter_num}.npy", start.best_field_reshaped)
        np.save(f"truth_field_{z}.npy", start.truth_field)
        np.save(f"data_field_{z}.npy", start.data)

    kvals, pspec_truth = bin_density_new(truth_field, 60, side_length)
    plt.loglog(kvals, pspec_truth, label="TRUTH", c="g", alpha=0.1)
    kvals, pspec = bin_density_new(np.reshape(start.s_field, (side_length, side_length)), 60, side_length)
    plt.loglog(kvals, pspec, label="initial guess", c="r", alpha=0.1)
    plt.legend(loc=1)

    # frame2 = fig_ps.add_axes((.1,.1,.8,.2))
    # difference = pspec_truth - pspec_best
    # plt.semilogx(kvals, difference, 'or', markersize=3)

    plt.title("power spectra")
    plt.savefig("all_pspec_alternate.png", dpi=300)

    # print("prior/likelihood")
    # print(prior_off)
    # print(likelihood_off)
    # data = np.load(f"battaglia plots/data_z_{z}_{side_length}.npy")
    # s_field = np.load(f"battaglia plots/guessed_field_z_{z}_{side_length}.npy")
    # truth_field = np.load(f"battaglia plots/truth_field_z_{z}_{side_length}.npy")

    # start with guessing density field with fixed bias

    # this stuff finds local min with parameter
    # print(f"latent space pixel value: {start.result[0]}")
    # print(f"actual pixel value: {start.data[0]}")
    # if param_going:
    #     start_2 = GD(actual_bias=actual_bias, param_init=param_init, curr_field=guessed_field, side_length=side_length,
    #                dimensions=dim, include_param=True)
    #     guessed_param = param_init = start_2.opt_result
    #     print(f"guessed param: {guessed_param}")
    #     print(f"latent space pixel value: {start.result[0,0]}")
    #     print(f"example pixel guess: {guessed_param * start.result[0,0]}")
    #     print(f"actual pixel value: {start_2.data[0,0]}")
    #     if start_2.final_likelihood_prior.primal > prev_likelihood:
    #         param_init = prev_param
    #         param_going = False
    #         start.check_field(start.truth_field, "truth field")
    #         start.check_field(start.result, "guessed field")
    #     else:
    #         prev_param = guessed_param
    #         prev_likelihood = start_2.final_likelihood_prior
    # ### Getting gradient at last theta
    # func = start_2.chi_sq_jax
    # latest_param = jnp.asarray([guessed_param])
    # chi_grad = grad(func)(latest_param)
    # print(chi_grad)
