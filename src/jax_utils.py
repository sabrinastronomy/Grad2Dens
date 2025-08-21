import jax.numpy as jnp
from jax import lax
from scipy import stats


def bias2zre_2d(dens, L, astro_params, zre):
    delta_dens = dens+1.
    delta_dens = jnp.where(jnp.isnan(delta_dens), 1., delta_dens)
    N = jnp.shape(delta_dens)[0]

    delta_k = jnp.fft.fftn(delta_dens, norm='ortho') \
        * jnp.power(L/N, jnp.ndim(delta_dens)/2)
    # PS_box = jnp.real(delta_k * jnp.conjugate(delta_k))
    # phases = jnp.where(
    #     jnp.abs(jnp.real(delta_k)) > 1e-14,
    #     jnp.arctan2(jnp.imag(delta_k), jnp.real(delta_k)),
    #     0.)

    # k-grid, units 1/Mpc
    k_x = jnp.fft.fftfreq(N, d=L/N/2./jnp.pi)
    kbox = jnp.sqrt(jnp.power(k_x, 2)[:, None] + jnp.power(k_x, 2))

    # return kbox
    b0, k0, alpha = astro_params
    # k0 = jnp.power(10, logk0)
    kbox_div = jnp.where(k0 > 1e-4, jnp.divide(kbox, k0), 0.)
    bmz = b0 / jnp.power(1. + kbox_div, alpha)
    zz_box = delta_k * bmz
    deltak_zre = zz_box
    deltaz = jnp.real(jnp.fft.ifftn(deltak_zre, norm='ortho'))
    deltaz = deltaz - jnp.mean(deltaz)
    zre_box = (1. + zre) * deltaz + zre

    return zre_box


def bias2brightness_2d(
        z, dens, L,
        astro_params, zre,
        cosmo_params, zslope=20.
        ):

    OMb, OMm, h = cosmo_params
    zre_box = bias2zre_2d(dens, L, astro_params, zre)
    xHI_box = (jnp.tanh((-zre_box+z)*zslope) + 1.)/2
    print(zslope)
    cosmo_prefac = 27. * OMb * h**2 / 0.023 *\
        jnp.sqrt(0.15/OMm/h**2) * jnp.sqrt((1.+z)/10.)
    dTb_box = cosmo_prefac * xHI_box * (1. + dens)

    return dTb_box, zre_box, xHI_box


def gaussian_prior(param, gp):

    mu, sigma = gp
    return (param - mu) ** 2 / sigma ** 2


def flat_prior(param, bounds):

    minp, maxp = bounds
    if (param < minp) or (param > maxp):
        return jnp.inf
    else:
        return 0.


def make_initial_density(N, L, ndim, pk_prior, verbose=False):

    k_x = (2. * jnp.pi) * jnp.fft.fftfreq(N, d=L/N)
    a = jnp.power(k_x, 2)[:, None] + jnp.power(k_x, 2)
    if ndim == 1:
        kbox = jnp.abs(k_x)
        size = (2, N)
    if ndim == 2:
        kbox = jnp.sqrt(a)
        size = (2, N, N)
    elif ndim == 3:
        kbox = jnp.sqrt(a[:, :, None] + jnp.power(k_x, 2))
        size = (2, N, N, N)
    powerbox = jnp.array(pk_prior(kbox))
    dx = L/N

    # gaussian_field
    means = jnp.zeros(kbox.shape)
    widths = jnp.sqrt(powerbox*0.5)
    a, b = jnp.random.normal(
        means,
        widths,
        size=size,
    )  # Mpc3
    if verbose:
        print('Second cumulant of real (imag) distribution is '
              f'{stats.kstat(a,2):.2e} ({stats.kstat(b, 2):.2e}).')
    u = jnp.fft.irfftn(
            (a + b * 1j),
            s=(kbox.shape),
            norm='ortho')
    u /= dx**(ndim/2)  # Mpc**ndim

    return u.real


def ps1d_2d(field, L, nbins=10):

    field = jnp.array(field)
    ndim = 2
    N = float(jnp.shape(field)[0])

    deltax = L/N
    maxk = 0.5 * 2.*jnp.pi/deltax  # Nyquist-Shannon

    # k-grid, units 1/Mpc
    k_x = jnp.fft.fftfreq(int(N), d=deltax/2./jnp.pi)
    k_norm = jnp.sqrt(jnp.power(k_x, 2)[:, None] + jnp.power(k_x, 2))

    # Fourier transform
    delta_k = jnp.fft.fftn(field, norm='ortho') * deltax**(ndim/2)
    # normalisation is 1/sqrt(N)
    # unit Mpc**(ndim/2)
    # power spectrum
    PS = jnp.real(delta_k*jnp.conjugate(delta_k))  # unit Mpc**ndim

    kbins = jnp.linspace(4.*jnp.pi/L, maxk, nbins)
    dk = jnp.diff(kbins).mean()
    kbin_edges = jnp.r_[kbins - dk / 2, kbins.max() + dk / 2]

    count = jnp.histogram(k_norm, kbin_edges)[0]
    count = jnp.array(count, dtype=float)
    k = jnp.histogram(
        k_norm,
        kbin_edges,
        weights=k_norm
        )[0]
    k /= count
    pk = jnp.histogram(
        k_norm,
        kbin_edges,
        weights=PS
        )[0]
    pk /= count
    pk2 = jnp.histogram(
        k_norm,
        kbin_edges,
        weights=PS**2
        )[0]
    pk2 /= count
    pk_err = jnp.sqrt(pk2-pk**2)/jnp.sqrt(count)

    return k, pk, pk_err


def get_dimless_ps_2d(field, L, nbins=10):

    k, ps, err = ps1d_2d(field, L, nbins)
    return k**2 * ps / 2. / jnp.pi