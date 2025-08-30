# MAP Density Field Inference (JAX + NumPyro)

Infrastructure to infer Maximum A Posteriori (MAP) estimates **and perform full Bayesian inference** of the **matter density field** from 21 cm brightness-temperature maps using:

* a differentiable forward model based on **Battaglia et al. (2013)**,
* gradient-based optimization with **jaxopt**, and
* **Hamiltonian Monte Carlo (HMC)** sampling via **NumPyro**.

> Author: **Sabrina Berger** · Started: **Nov 2022**
> Language: Python (tested with **3.9**) · Backend: **JAX**, **NumPyro**

---

## Contents

* [Overview](#overview)
* [Features](#features)
* [Install](#install)
* [Quickstart](#quickstart)
* [Bayesian (HMC/NUTS) with NumPyro](#bayesian-hmcnuts-with-numpyro)
* [Outputs](#outputs)
* [API Reference](#api-reference)

  * [ConfigParam](#configparam)
  * [SwitchMinimizer](#switchminimizer)
  * [InferDens](#inferdens)
  * [grid\_test](#grid_test)
* [Priors & Likelihood](#priors--likelihood)
* [Notes & Gotchas](#notes--gotchas)
* [Citing](#citing)

---

## Overview

The package reconstructs the latent matter density field $s$ by:

* **MAP inference:** minimizing a posterior objective
* **Bayesian inference:** sampling the full posterior using NumPyro’s HMC/NUTS

The forward model $f$ maps density to **brightness temperature** via the Battaglia+2013 prescription (class `Dens2bBatt`).

## Features

* **JAX** implementation with **jaxopt.LBFGS** for fast, differentiable optimization.
* **NumPyro HMC/NUTS** posterior sampling (full-Bayes) with differentiable forward model.
  – Works with diagonal Gaussian likelihoods or user-supplied covariances; supports masking subsets of pixels in the likelihood.
* **Forward model:** Battaglia et al. (2013) bias mapping (via `Dens2bBatt`).
* **Instrument model:** optional SKA-like effects via `SKAEffects`.
* **Priors:**

  * *Old prior:* independent Fourier-mode Gaussian prior (2D; real/imag parts).
  * *New prior:* binned power-spectrum prior in $k$-space (2D/3D) or theory-anchored variant.
* **Dimensionality:** 2D and 3D boxes (most plotting utilities assume 2D or 2D slices of 3D).
* **Plotting suite:** power spectra, iteration panels, residuals, correlation and histograms, pixel masks.

## Install

```bash
pip install jax jaxlib jaxopt numpy matplotlib scipy powerbox numpyro
```

## Quickstart

### MAP Inference

See [InferDens](#inferdens) usage in [Quickstart](#quickstart).

### Bayesian Inference (NumPyro)

Define a generative model in NumPyro:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_battaglia_full import Dens2bBatt

sigma_th = jnp.std(data)
dims = (32, 32)  # shape of density field

# Simple Gaussian field prior model
def bayesian_model(obs=None):
    density = numpyro.sample(
        "theory",
        dist.Normal(0.0, 3).expand(dims).to_event(len(dims))
    )
    batt_model_instance = Dens2bBatt(
        density,
        resolution=resolution, set_z=z,
        physical_side_length=physical_side_length, flow=True,
        free_params=fiducial_params,
        apply_ska=config.ska_effects,
    )
    inferred_data = batt_model_instance.temp_brightness
    numpyro.sample("data", dist.Normal(inferred_data, sigma_th), obs=obs)
```

Run HMC:

```python
from numpyro.infer import MCMC, NUTS

kernel = NUTS(bayesian_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), obs=data)
posterior_samples = mcmc.get_samples()
```

You can also implement *old-style priors* (e.g. Fourier or power-spectrum penalties) with `numpyro.factor("power_spectrum_prior", log_prior)`.

## Bayesian (HMC/NUTS) with NumPyro

This repo can also **sample the posterior** over the density field using **NumPyro** with a differentiable forward model.

> Install extras: `pip install numpyro`

**Model pieces**

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_battaglia_full import Dens2bBatt
from theory_matter_ps import circular_spec_normal  # or spherical for 3D

# shapes and constants
dims = (side_length, side_length)            # or (D, H, W) for 3D
resolution = physical_side_length / side_length
area = side_length**2
sigma_th = jnp.std(DensityBlock.data)        # diag noise level (example)

# (optional) reference theory P(k) for priors
counts2d, truth_final_pspec, k_bins = circular_spec_normal(
    truth_field, bins, resolution, area
)

def bayesian_model(obs=None):
    # zero-mean Normal prior on the field (elementwise), with event dims set
    density = numpyro.sample(
        "theory",
        dist.Normal(0.0, 3.0).expand(dims).to_event(len(dims))
    )
    batt = Dens2bBatt(
        density, resolution=resolution, set_z=z,
        physical_side_length=physical_side_length, flow=True,
        free_params=fiducial_params,
        apply_ska=DensityBlock.config_params.ska_effects,
    )
    inferred_data = batt.temp_brightness
    numpyro.sample("data", dist.Normal(inferred_data, sigma_th), obs=obs)
```

**Older variant (elementwise sample then reshape)**

```python
def old_bayesian_model(obs=None):
    density = numpyro.sample("theory", dist.Normal(0, 5).expand(dims))
    density = jnp.reshape(density, dims)
    batt = Dens2bBatt(
        density, resolution=resolution, set_z=z,
        physical_side_length=physical_side_length, flow=True,
        free_params=fiducial_params,
        apply_ska=DensityBlock.config_params.ska_effects,
    )
    inferred_data = batt.temp_brightness
    numpyro.sample("data", dist.Normal(inferred_data, sigma_th), obs=obs)
```

**Optional power-spectrum prior (NumPyro factor)**

```python
# inside the model, after computing `density`
counts2d, pspec2d, k_bins = circular_spec_normal(density, bins, resolution, area)
log_prior = -0.5 * jnp.sum(
    (jnp.log10(truth_final_pspec + 1e-10) - jnp.log10(pspec2d + 1e-10))**2
)
weight = 1.0  # tune
numpyro.factor("power_spectrum_prior", weight * log_prior)
```

**Run NUTS**

```python
from numpyro.infer import MCMC, NUTS

kernel = NUTS(bayesian_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), obs=observed_Tb)
posterior = mcmc.get_samples()
```

**Masked likelihood (use only ionized pixels)**

```python
# Suppose `ionized_mask` is a boolean mask over `dims`
def bayesian_model_mask_neutral(obs=None, ionized_mask=ionized_mask):
    density = numpyro.sample("theory", dist.Normal(0, 10).expand(dims).to_event(len(dims)))
    batt = Dens2bBatt(density, resolution=resolution, set_z=z,
                      physical_side_length=physical_side_length, flow=True,
                      free_params=fiducial_params,
                      apply_ska=DensityBlock.config_params.ska_effects)
    full_Tb = batt.temp_brightness
    masked_Tb = full_Tb[ionized_mask]
    numpyro.sample("data", dist.Normal(masked_Tb, sigma_th), obs=obs[ionized_mask])
```

> Notes: (i) For non-diagonal noise, replace the Normal with a `MultivariateNormal` and supply `covariance_matrix=cov_matrix_Tb` (be mindful of dimensionality); (ii) using `.to_event(len(dims))` tells NumPyro the field is a single random tensor draw (not iid across pixels).

## Outputs

* MAP outputs: same as before (plots/arrays saved per iteration).
* HMC outputs: posterior samples of the density field (shape `[num_samples, *dims]`), accessible via `mcmc.get_samples()`.

---

## API Reference

### `ConfigParam`

*(unchanged, see previous section)*

### `SwitchMinimizer`

*(unchanged)*

### `InferDens`

*(unchanged, see previous section)*

### `grid_test`

*(unchanged)*

### Bayesian Models

Two reference implementations provided:

* `bayesian_model`: Gaussian prior on density, Gaussian likelihood on Tb.
* `old_bayesian_model`: alternative Normal prior (sigma=5) with optional power-spectrum prior.
* Masked versions allow restricting likelihood to ionized pixels.

Use with NumPyro inference APIs (`NUTS`, `HMC`, `SVI`).

---

## Priors & Likelihood

Now includes **NumPyro factors** for power-spectrum priors and the ability to mask likelihood terms.

---

## Notes & Gotchas

* HMC over full density fields can be memory intensive; start with small `dims`.
* Ensure `sigma_th` matches noise level (default: std of data).
* Power-spectrum priors can be added via `numpyro.factor`.
* Posterior sampling requires careful tuning of `num_warmup` and `step_size`.

---

## Citing

If you use the Bayesian NumPyro functionality, please cite **NumPyro (Phan et al. 2019)** in addition to Battaglia+2013 and this repository.
