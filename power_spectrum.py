from alternating import InferDens

samp = InferDens(seed=None, z=8, num_bins=256, iter_num_max=1000, run_optimizer=False, plot_direc="new_seed_1011_256_bins", pspec_on_plot=True)
samp.plot_pspec_and_panel(normalize=False)

# samp = InferDens(z=8, iter_num_max=1000, run_optimizer=False, plot_direc="data_adversarial_classic_prior")
# samp.plot_pspec_and_panel(normalize=True)

# samp.plot_2_mse()