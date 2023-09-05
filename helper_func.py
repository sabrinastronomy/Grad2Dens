import numpy as np

def bin_density_new(field, nbins, side_length, bin=True):
    fft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(field)))
    fft_data_squared = np.abs(fft_data) ** 2
    k_arr = np.fft.fftshift(np.fft.fftfreq(side_length)) * 2 * np.pi
    k1, k2 = np.meshgrid(k_arr, k_arr)
    k_mag_full = np.sqrt(k1 ** 2 + k2 ** 2)
    k_mag = np.asarray(k_mag_full.flatten())
    if not bin:
        return k_mag.flatten(), fft_data.flatten()
    Abins, bin_edges = np.histogram(k_mag, nbins)
    binned_power = np.zeros(nbins)
    count = np.zeros(nbins)
    for i in range(side_length):
        for j in range(side_length):
            k_mag_at_this_index_2 = k_mag_full[i,j]
            for bin_num in range(nbins):
                if bin_edges[bin_num] <= k_mag_at_this_index_2 < bin_edges[bin_num + 1]:
                    binned_power[bin_num] = binned_power[bin_num] + fft_data_squared[i, j]
                    # binned_power_masoud[bin_num] += fft_data_squared[i, j] * k_mag_at_this_index_2
                    # w_rings[bin_num] += k_mag_at_this_index_2
                    count[bin_num] = count[bin_num] + 1
                    break
    # pspec = binned_power/count
    # binned_power *= np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
    kvals = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pspec = binned_power / count / (side_length**2)
    return kvals, pspec