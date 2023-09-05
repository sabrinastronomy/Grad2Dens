from alternating import InferDens

samp = InferDens(z=8, num_bins=1000, iter_num_max=1000, run_optimizer=False, plot_direc="data_adversarial_new_prior_2", pspec_on_plot=True)
samp.plot_pspec_and_panel(normalize=False)

# samp = InferDens(z=8, iter_num_max=1000, run_optimizer=False, plot_direc="data_adversarial_classic_prior")
# samp.plot_pspec_and_panel(normalize=True)

# samp.plot_2_mse()