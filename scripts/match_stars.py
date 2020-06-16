#!/usr/bin/env python

from __future__ import print_function, division


import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import astropy.units as u
from astropy.coordinates import SkyCoord

from glob import glob
import os


def load_data(fnames):
    # Load in all the data files
    d = []

    for fn in fnames:
        print('Loading {:s} ...'.format(fn))
        with h5py.File(fn, 'r') as f:
            d.append(f['stellar_phot_spec_ast'][:])
    
    return np.hstack(d)


def find_ddpayne_repeats(d):
    # Sort by longitude, then latitude
    idx_sort = np.lexsort((d['gal_l'], d['gal_b']))
    
    # Apply sort
    d_s = d[idx_sort]
    
    # Find indices where lon or lat change
    idx_split = (np.diff(d_s['gal_l']) > 0.) | (np.diff(d_s['gal_b']) > 0)
    idx_split = np.hstack([0, np.where(idx_split)[0] + 1, len(d)])
    
    # Go through each unique object
    for i0,i1 in zip(idx_split[:-1],idx_split[1:]):
        if i1 > i0+1:
            print(d_s[i0:i1])
            print('')
    
    # Find unique longitudes
    #lon, idx, n = np.unique(d['gal_l'], return_index=True, return_counts=True)


def match_catalogs(d1, d2, max_sep=0.1*u.arcsec):
    c1 = SkyCoord(d1['gal_l']*u.deg, d1['gal_b']*u.deg, frame='galactic')
    c2 = SkyCoord(d2['gal_l']*u.deg, d2['gal_b']*u.deg, frame='galactic')
    idx_c1, sep, _ = c2.match_to_catalog_sky(c1)
    
    idx = (sep < max_sep)
    idx_c1 = idx_c1[idx]
    
    return idx_c1, idx, sep[idx]


def match_3_catalogs(d1, d2, d3, max_sep=0.1*u.arcsec):
    c1 = SkyCoord(d1['gal_l']*u.deg, d1['gal_b']*u.deg, frame='galactic')
    c2 = SkyCoord(d2['gal_l']*u.deg, d2['gal_b']*u.deg, frame='galactic')
    c3 = SkyCoord(d3['gal_l']*u.deg, d3['gal_b']*u.deg, frame='galactic')
    
    def match_pair(x,y):
        idx_x, sep, _ = y.match_to_catalog_sky(x)
        idx = (sep < max_sep)
        idx_xy = (idx_x[idx], np.where(idx)[0])
        xy = x[idx_xy[0]]
        return xy, idx_xy
    
    def filter_arrs(a_list, idx_remove):
        idx_keep = np.ones(a_list[0].size, dtype='bool')
        idx_keep[idx_remove] = 0
        return [a[idx_keep] for a in a_list]
    
    # Match catalogs pairwise
    c12, idx_12 = match_pair(c1, c2)
    c23, idx_23 = match_pair(c2, c3)
    c13, idx_13 = match_pair(c1, c3)
    
    print(f'{len(idx_12[0])} 1&2 matches.')
    print(f'{len(idx_13[0])} 1&3 matches.')
    print(f'{len(idx_23[0])} 2&3 matches.')
    
    # Match 12 & 13
    c123, idx_12_13 = match_pair(c12, c13)
    print(idx_12_13)
    idx_123 = (
        idx_12[0][idx_12_13[0]], # indices in d1 of 123 matches
        idx_12[1][idx_12_13[0]], # indices in d2 of 123 matches
        idx_13[1][idx_12_13[1]]  # indices in d3 of 123 matches
    )
    print(f'{len(idx_123[0])} 1&2&3 matches.')
    # Filter matches out of 13
    idx_13 = filter_arrs(idx_13, idx_12_13[1])
    #idx_keep = np.ones(idx_13.size, dtype='bool')
    #idx_keep[idx_12_13[1]] = 0
    #idx_13 = idx_13[idx_keep]
    
    # Match 12 & 23
    c123, idx_12_23 = match_pair(c12, c23)
    # Filter matches out of 12 and 23
    idx_12 = filter_arrs(idx_12, idx_12_23[0])
    idx_23 = filter_arrs(idx_23, idx_12_23[1])
    
    # Concatenate 12, 13, 23, 123
    idx1 = np.hstack([
        idx_12[0],
        idx_13[0],
        np.full_like(idx_23[0], -1),
        idx_123[0]
    ])
    idx2 = np.hstack([
        idx_12[1],
        np.full_like(idx_13[0], -1),
        idx_23[0],
        idx_123[1]
    ])
    idx3 = np.hstack([
        np.full_like(idx_12[0], -1),
        idx_13[1],
        idx_23[1],
        idx_123[2]
    ])
    
    return idx1, idx2, idx3


def main():
    ext = 'pdf'
    fig_dir = 'plots'
    
    d_galah = load_data(glob('data/galah_data_*to*.h5'))
    d_apogee = load_data(glob('data/dr16_data_*to*.h5'))
    d_ddpayne = load_data(glob('data/ddpayne_data_*to*.h5'))
    
    # Match GALAH, DDPayne and APOGEE catalogs
    idx_galah, idx_ddpayne, idx_apogee = match_3_catalogs(
        d_galah, d_ddpayne, d_apogee
    )
    d_galah = d_galah[idx_galah]
    d_ddpayne = d_ddpayne[idx_ddpayne]
    d_apogee = d_apogee[idx_apogee]
    mask_galah = (idx_galah != -1)
    mask_ddpayne = (idx_ddpayne != -1)
    mask_apogee = (idx_apogee != -1)
    print('{:d} matches.'.format(len(d_galah)))
    
    # Save matches
    fname = 'data/crossmatches_galah_apogee_ddpayne.h5'
    dset_kw = dict(chunks=True, compression='gzip', compression_opts=3)
    with h5py.File(fname, 'w') as f:
        f.create_dataset('/galah', data=d_galah, **dset_kw)
        f.create_dataset('/ddpayne', data=d_ddpayne, **dset_kw)
        f.create_dataset('/apogee', data=d_apogee, **dset_kw)
        f.create_dataset('/mask_galah', data=mask_galah.astype('u1'), **dset_kw)
        f.create_dataset('/mask_ddpayne', data=mask_ddpayne.astype('u1'), **dset_kw)
        f.create_dataset('/mask_apogee', data=mask_apogee.astype('u1'), **dset_kw)
    
    #return 0
    
    # Load matches
    print('Loading matches ...')
    fname = 'data/crossmatches_galah_apogee_ddpayne.h5'
    with h5py.File(fname, 'r') as f:
        d_galah = f['/galah'][:]
        d_ddpayne = f['/ddpayne'][:]
        d_apogee = f['/apogee'][:]
        
        mask_galah = (f['/mask_galah'][:].astype('bool'))
        mask_ddpayne = (f['/mask_ddpayne'][:].astype('bool'))
        mask_apogee = (f['/mask_apogee'][:].astype('bool'))
    
    print(f'{np.count_nonzero(mask_galah)} GALAH sources.')
    print(f'{np.count_nonzero(mask_ddpayne)} LAMOST sources.')
    print(f'{np.count_nonzero(mask_apogee)} APOGEE sources.')
    
    # Extract labels from each survey
    params = {
        'teff': {
            'galah': d_galah['teff'],
            'apogee': d_apogee['sdss_aspcap_param'][:,0],
            'lamost': d_ddpayne['ddpayne_teff']
        },
        'logg': {
            'galah': d_galah['logg'],
            'apogee': d_apogee['sdss_aspcap_param'][:,1],
            'lamost': d_ddpayne['ddpayne_logg']
        },
        'feh': {
            'galah': d_galah['feh'],
            'apogee': d_apogee['sdss_aspcap_param'][:,3],
            'lamost': d_ddpayne['ddpayne_feh']
        }
    }
    param_errs = {
        'teff': {
            'galah': d_galah['teff_err'],
            'apogee': d_apogee['sdss_aspcap_teff_err'],
            'lamost': d_ddpayne['ddpayne_teff_err']
        },
        'logg': {
            'galah': d_galah['logg_err'],
            'apogee': d_apogee['sdss_aspcap_logg_err'],
            'lamost': d_ddpayne['ddpayne_logg_err']
        },
        'feh': {
            'galah': d_galah['feh_err'],
            'apogee': d_apogee['sdss_aspcap_m_h_err'],
            'lamost': d_ddpayne['ddpayne_feh_err']
        }
    }
    survey_masks = {
        'galah': mask_galah,
        'apogee': mask_apogee,
        'lamost': mask_ddpayne
    }
    
    # Residuals in each label
    param_dlims = {
        'teff': (-500., 500.),
        'logg': (-0.5, 0.5),
        'feh': (-0.5, 0.5)
    }
    
    # Labels
    param_names = ('teff', 'logg', 'feh')

    param_labels = {
        'teff': r'T_{\mathrm{eff}}',
        'logg': r'\log \left( g \right)',
        'feh': r'\left[ \mathrm{Fe} / \mathrm{H} \right]'
    }
    survey_labels = {
        'galah': r'\mathrm{GALAH}',
        'apogee': r'\mathrm{APOGEE}',
        'lamost': r'\mathrm{LAMOST}'
    }
    
    # Choose one survey to anchor the (teff, logg, feh) scale
    surveys = ('apogee', 'galah', 'lamost')
    
    # Plot histogram of residuals
    print('Plotting histograms of residuals ...')
    for name in param_names:
        fig,ax_list = plt.subplots(3,2, figsize=(12,13.5))
        fig.suptitle(rf'$\Delta {param_labels[name]}$')
        
        #for ax,comp in zip(ax_list, comparisons):
        for (ax1,ax2),(k1,k2) in zip(ax_list,((0,2),(1,2),(0,1))):
            comp, anchor = surveys[k1], surveys[k2]
            
            dval = params[name][comp] - params[name][anchor]
            var = (param_errs[name][comp])**2 + (param_errs[name][anchor])**2
            
            idx = (
                np.isfinite(dval) &
                survey_masks[comp] &
                survey_masks[anchor]
            )
            print(f'{comp} & {anchor}: {np.count_nonzero(idx)} matches.')
            dval = dval[idx]
            var = var[idx]
            
            idx_chi = np.ones(dval.size, dtype='bool')
            for i in range(3):
                delta_est = (
                    np.mean(dval[idx_chi] / var[idx_chi])
                    / np.mean(1./var[idx_chi])
                )
                chi = (dval - delta_est) / np.sqrt(var)
                idx_chi = (np.abs(chi) < 3.)
                print(f'  -> {1.-np.count_nonzero(idx_chi)/idx_chi.size:.3f} rejected.')
                print(f'Delta = {delta_est:.5f}')
            
            ax1.hist(dval, range=param_dlims[name], bins=100, alpha=0.7)
            ax1.axvline(np.median(dval), c='g', lw=2.0, alpha=0.7)
            ax1.axvline(np.median(delta_est), c='orange', lw=2.0, alpha=0.7)
            ax1.grid('on', alpha=0.25)
            ax1.xaxis.set_major_locator(ticker.AutoLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            txt = (
                r'$'
                f'{survey_labels[comp]} - {survey_labels[anchor]}'
                f' = {delta_est:.5f}'
                r'$'
            )
            ax1.text(
                0.03, 0.95,
                txt,
                ha='left', va='top',
                transform=ax1.transAxes
            )
            
            _,x_edges,_ = ax2.hist(chi, range=(-5.,5.), bins=100, alpha=0.7)
            
            dx = x_edges[1] - x_edges[0]
            x_gauss = np.linspace(-5., 5., 1000)
            y_gauss = dx * chi.size * np.exp(-0.5*x_gauss**2) / np.sqrt(2.*np.pi)
            ax2.plot(x_gauss, y_gauss, c='g', alpha=0.5)
            
            chi2_mean = np.mean(chi[np.abs(chi)<5.]**2)
            txt = (
                r'$'
                r'\langle \chi^2 \rangle = '
                f'{chi2_mean:.3f}'
                r'$'
            )
            ax2.text(
                0.03, 0.95, txt,
                ha='left', va='top',
                transform=ax2.transAxes
            )
            
            ax2.grid('on', alpha=0.25)
            ax2.xaxis.set_major_locator(ticker.AutoLocator())
            ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        for ax in ax_list.flat[:-2]:
            ax.set_xticklabels([])
        for ax in ax_list.flat:
            ax.set_yticklabels([])
        
        ax_list[2,0].set_xlabel(rf'$\Delta {param_labels[name]}$')
        ax_list[2,1].set_xlabel(rf'$\chi \left( {param_labels[name]} \right)$')
        
        fig.subplots_adjust(
            top=0.94, bottom=0.06,
            left=0.05, right=0.95,
            hspace=0.05, wspace=0.05
        )
        
        fig.savefig(os.path.join(fig_dir, f'resid_hist_{name}.{ext}'), dpi=150)
        plt.close(fig)
    
    return 0


if __name__ == '__main__':
    main()
