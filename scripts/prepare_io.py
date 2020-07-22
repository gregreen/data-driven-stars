#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py
from glob import glob


def get_corr_matrix(cov):
    rho = cov.copy()
    sqrt_cov_diag = np.sqrt(cov[np.diag_indices(cov.shape[0])])
    rho /= sqrt_cov_diag[:,None]
    rho /= sqrt_cov_diag[None,:]
    rho[np.diag_indices(cov.shape[0])] = sqrt_cov_diag
    return rho


def load_data(fnames):
    # Load in all the data files
    d = []
    b19 = []
    b19_err = []

    for fn in fnames:
        print('Loading {:s} ...'.format(fn))
        with h5py.File(fn, 'r') as f:
            d.append(f['stellar_phot_spec_ast'][:])
            b19.append(f['reddening'][:])
            b19_err.append(f['reddening_err'][:])

    d = np.hstack(d)
    b19 = np.hstack(b19)
    b19_err = np.hstack(b19_err)

    return d, b19, b19_err


def extract_data(d, b19, b19_err):
    # Gather into one dataset
    dtype = [
        ('atm_param', '3f4'),
        ('atm_param_cov', 'f4', (3,3)),
        ('atm_param_p', '3f4'),           # normalized
        ('atm_param_cov_p', 'f4', (3,3)), # normalized
        ('r', 'f4'),
        ('r_err', 'f4'),
        ('mag', '13f4'),
        ('mag_err', '13f4'),
        ('parallax', 'f4'),
        ('parallax_err', 'f4'),
        ('atm_source', 'S6'),
        ('r_source', 'S7')
    ]
    io_data = np.empty(d.size, dtype=dtype)
    
    # Offsets to bring spectroscopic labels from different
    # surveys into alignment
    offsets = {
        'apogee': np.array([23.04715, 0.01189, 0.05019]),
        'lamost': np.array([0., 0., 0.]),
        'galah': np.array([-3.60096, -0.01396, 0.06770])
    }
    # Fix offsets to GALAH
    offsets['apogee'] -= offsets['galah']
    offsets['lamost'] -= offsets['galah']
    offsets['galah'][:] = 0.
    
    print('offsets:')
    for key in offsets:
        print(f'  * {key}: {offsets[key]}')

    # How to load data depends on survey
    if 'sdss_aspcap_param' in d.dtype.names: # APOGEE
        io_data['atm_source'] = 'apogee'
        
        param_idx = [0, 1, 3] # (T_eff, logg, [M/H])
        param_name = ['teff', 'logg', 'm_h']

        # Copy in parameters and corresponding covariance entries
        for k,i in enumerate(param_idx):
            io_data['atm_param'][:,k] = d['sdss_aspcap_param'][:,i]
            for l,j in enumerate(param_idx):
                io_data['atm_param_cov'][:,k,l] = d['sdss_aspcap_fparam_cov'][:,9*k+l]
        
        io_data['atm_param'] -= offsets['apogee'][None,:]

        # Copy calibrated errors into diagonals of covariance matrices.
        #   - Keep uncalibrated errors if larger.
        for k,n in enumerate(param_name):
            io_data['atm_param_cov'][:,k,k] = np.maximum(
                d[f'sdss_aspcap_{n}_err']**2,
                io_data['atm_param_cov'][:,k,k]
            )
        
        # Scale covariance entries according to ratio of
        # calibrated to uncalibrated errors.
        #err_ratio = [
        #    d[f'sdss_aspcap_{n}_err']/np.sqrt(d['sdss_aspcap_fparam_cov'][:,9*i+i])
        #    for n,i in zip(param_name, param_idx)
        #]
        #for i in range(3):
        #    #n = param_name[i]
        #    #k = param_idx[i]
        #    #print(err_ratio[i])
        #    #print(d[f'sdss_aspcap_{n}_err'])
        #    #print(np.sqrt(d['sdss_aspcap_fparam_cov'][:,9*k+k]))
        #    #print('')
        #    io_data['atm_param_cov'][:,i,:] *= err_ratio[i][:,None]
        #    io_data['atm_param_cov'][:,:,i] *= err_ratio[i][:,None]
    elif 'ddpayne_teff' in d.dtype.names: # LAMOST DDPAYNE
        io_data['atm_source'] = 'lamost'
        
        # Copy in parameters
        io_data['atm_param'][:,0] = d['ddpayne_teff'][:]
        io_data['atm_param'][:,1] = d['ddpayne_logg'][:]
        io_data['atm_param'][:,2] = d['ddpayne_feh'][:]
        io_data['atm_param'] -= offsets['lamost'][None,:]

        # Diagonal covariance matrix
        io_data['atm_param_cov'][:] = 0.
        io_data['atm_param_cov'][:,0,0] = d['ddpayne_teff_err']**2.
        io_data['atm_param_cov'][:,1,1] = d['ddpayne_logg_err']**2.
        io_data['atm_param_cov'][:,2,2] = d['ddpayne_feh_err']**2.
    elif 'snr_c1' in d.dtype.names: # GALAH
        io_data['atm_source'] = 'galah'
        
        # Copy in parameters
        io_data['atm_param'][:,0] = d['teff'][:]
        io_data['atm_param'][:,1] = d['logg'][:]
        io_data['atm_param'][:,2] = d['feh'][:]
        io_data['atm_param'] -= offsets['galah'][None,:]

        # Diagonal covariance matrix
        io_data['atm_param_cov'][:] = 0.
        io_data['atm_param_cov'][:,0,0] = d['teff_err']**2.
        io_data['atm_param_cov'][:,1,1] = d['logg_err']**2.
        io_data['atm_param_cov'][:,2,2] = d['feh_err']**2.
    
    # Add in error floor to atmospheric parameters
    sigma_atm_param_floor = [10., 0.05, 0.03] # (T_eff, logg, [M/H])
    for i,sig in enumerate(sigma_atm_param_floor):
        io_data['atm_param_cov'][:,i,i] += sig**2

    # Print correlation matrices, for fun
    for i in range(10):
        rho = get_corr_matrix(io_data['atm_param_cov'][i])
        print('Correlation matrices:')
        print(np.array2string(
            rho,
            formatter={'float_kind':lambda z:'{: >7.4f}'.format(z)}
        ))

    # Reddening sources, in order of priority:
    #   1. If |z| > 400 pc: Use SFD with 10% uncertainty
    #   2. If parallax/error > 5: Use Bayestar19
    #   3. Otherwise: Use 0 +- SFD
    
    z_0 = 0.4 # kpc
    sin_b_over_z = np.abs(np.sin(np.radians(d['gal_b']))) / z_0
    idx_z = (d['parallax'] + 5*d['parallax_err'] < sin_b_over_z)
    idx_plx_over_err = (d['parallax'] / d['parallax_err'] > 5.)
    idx_b19 = np.isfinite(b19)
    
    idx_sfd = idx_z
    idx_b19 = ~idx_sfd & idx_plx_over_err & idx_b19
    idx_default = ~idx_sfd & ~idx_b19
    
    print(r'Reddening sources:')
    print(r' * SFD: {:.4f}'.format(np.count_nonzero(idx_sfd)/idx_sfd.size))
    print(r' * B19: {:.4f}'.format(np.count_nonzero(idx_b19)/idx_b19.size))
    print(r' * ---: {:.4f}'.format(np.count_nonzero(idx_default)/idx_default.size))
    
    r_err_scale = 0.1
    
    io_data['r'][idx_default] = 0.
    io_data['r_err'][idx_default] = d['SFD'][idx_default]
    io_data['r_source'][idx_default] = 'default'
    
    #idx = idx_plx_over_err & idx_b19
    b19_val = b19[idx_b19]
    b19_err = b19_err[idx_b19]
    b19_err = np.sqrt(b19_err**2 + r_err_scale**2*b19_val**2)
    io_data['r'][idx_b19] = b19_val
    io_data['r_err'][idx_b19] = b19_err
    io_data['r_source'][idx_b19] = 'b19'
    
    io_data['r'][idx_sfd] = d['SFD'][idx_sfd]
    io_data['r_err'][idx_sfd] = 0.1 * d['SFD'][idx_sfd]
    io_data['r_source'][idx_sfd] = 'sfd'
    
    # Add in reddening error floor
    r_err_floor = 0.02
    io_data['r_err'] = np.sqrt(
          io_data['r_err']**2
        + r_err_floor**2
        #+ (r_err_scale*io_data['r'])**2
    )
    
    ## Use Bayestar19 reddening by default
    #io_data['r'] = b19[:]
    #io_data['r_err'] = b19_err[:]
    #
    ## Use SFD reddening as fallback
    #idx = ~np.isfinite(b19)
    #io_data['r'][idx] = d['SFD'][idx]
    #io_data['r_err'][idx] = d['SFD'][idx]
    
    # Stricter fracflux cut on WISE passbands
    idx = (d['unwise_fracflux'] < 0.5)
    d['unwise_mag'][idx] = np.nan
    d['unwise_mag_err'][idx] = np.nan

    # Copy in magnitudes
    io_data['mag'][:,0] = d['gaia_g_mag']
    io_data['mag_err'][:,0] = d['gaia_g_mag_err']
    io_data['mag'][:,1] = d['gaia_bp_mag']
    io_data['mag_err'][:,1] = d['gaia_bp_mag_err']
    io_data['mag'][:,2] = d['gaia_rp_mag']
    io_data['mag_err'][:,2] = d['gaia_rp_mag_err']
    io_data['mag'][:,3:8] = d['ps1_mag']
    io_data['mag_err'][:,3:8] = d['ps1_mag_err']
    for i,b in enumerate('JHK'):
        io_data['mag'][:,8+i] = d[f'tmass_{b}_mag']
        io_data['mag_err'][:,8+i] = d[f'tmass_{b}_mag_err']
    io_data['mag'][:,11:13] = d['unwise_mag']
    io_data['mag_err'][:,11:13] = d['unwise_mag_err']

    io_data['parallax'][:] = d['parallax']
    io_data['parallax_err'][:] = d['parallax_err']

    # Add in photometric error floors
    mag_err_floor = 0.02 * np.ones(13)
    io_data['mag_err'] = np.sqrt(
          io_data['mag_err']**2
        + mag_err_floor[None,:]**2
    )

    # Filter out magnitudes with err > 0.2
    idx = (io_data['mag_err'] > 0.2)
    io_data['mag'][idx] = np.nan
    io_data['mag_err'][idx] = np.nan

    return io_data


def extract_data_multiple(fname_lists):
    d_list = []

    for fnames in fname_lists:
        d,b19,b19_err = load_data(fnames)
        d = extract_data(d, b19, b19_err)
        print(f'Extracted {d.size} stars.')
        d_list.append(d)

    d = np.hstack(d_list)
    
    return d


def finalize_data(d):
    # Cuts on atmospheric parameters
    err_max = [200., 0.5, 0.5] # (T_eff, logg, [M/H])
    idx = np.ones(d.size, dtype='bool')
    for i,emax in enumerate(err_max):
        idx &= (
              np.isfinite(d['atm_param'][:,i])
            & np.isfinite(d[f'atm_param_cov'][:,i,i])
            & (d[f'atm_param_cov'][:,i,i] < emax*emax)
        )
    print(f'Filtered out {np.count_nonzero(~idx)} stars based on '
           'atmospheric parameters.')
    d = d[idx]

    # Normalize atmospheric parameters
    atm_param_med = np.median(d['atm_param'], axis=0)
    atm_param_std = np.std(d['atm_param'], axis=0)
    d['atm_param_p'] = (
        (d['atm_param'] - atm_param_med[None,:]) / atm_param_std[None,:]
    )
    d['atm_param_cov_p'][:] = d['atm_param_cov'][:]
    for i in range(3):
        d['atm_param_cov_p'][:,i,:] /= atm_param_std[i]
        d['atm_param_cov_p'][:,:,i] /= atm_param_std[i]

    return d, (atm_param_med, atm_param_std)


def print_stats(d):
    n_d = d.size
    
    print('Atmospheric parameter source:')
    for key in np.unique(d['atm_source']):
        n = np.count_nonzero(d['atm_source'] == key)
        print(f'  * {key.decode("utf-8")} : {n} ({n/n_d:.3f})')
    
    print('Reddening source:')
    for key in np.unique(d['r_source']):
        n = np.count_nonzero(d['r_source'] == key)
        print(f'  * {key.decode("utf-8")} : {n} ({n/n_d:.3f})')
    
    print('Sources per band:')
    n_pi = np.count_nonzero(
          np.isfinite(d['parallax'])
        & (d['parallax'] / d['parallax_err'] > 5.)
    )
    print(f'  *  pi : {n_pi} ({n_pi/n_d:.3f})')
    
    n_band = np.count_nonzero(np.isfinite(d['mag']), axis=0)
    bands = ['G','BP','RP'] + list('grizyJH') + ['K_s','W_1','W_2']
    for b,n in zip(bands,n_band):
        print(f'  * {b: >3s} : {n} ({n/n_d:.3f})')


def main():
    fnames = [
        glob('data/dr16_data_*to*.h5'),
        glob('data/ddpayne_data_*to*.h5'),
        glob('data/galah_data_*to*.h5')
    ]
    d = extract_data_multiple(fnames)
    d,(atm_param_med,atm_param_std) = finalize_data(d)
    print_stats(d)

    with h5py.File('data/apogee_lamost_galah_data.h5', 'w') as f:
        dset = f.create_dataset(
            'io_data',
            data=d,
            chunks=True,
            compression='gzip',
            compression_opts=3
        )
        dset.attrs['atm_param_med'] = atm_param_med
        dset.attrs['atm_param_std'] = atm_param_std

    return 0


if __name__ == '__main__':
    main()

