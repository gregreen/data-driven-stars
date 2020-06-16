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
    fig_dir = '/n/fink2/www/ggreen/dd_stellar_models/'
    
    #d_galah = load_data(glob('data/galah_data_*to*.h5'))
    #d_apogee = load_data(glob('data/dr16_data_*to*.h5'))
    #d_ddpayne = load_data(glob('data/ddpayne_data_*to*.h5'))
    #
    ## Match GALAH, DDPayne and APOGEE catalogs
    #idx_galah, idx_ddpayne, idx_apogee = match_3_catalogs(
    #    d_galah, d_ddpayne, d_apogee
    #)
    #d_galah = d_galah[idx_galah]
    #d_ddpayne = d_ddpayne[idx_ddpayne]
    #d_apogee = d_apogee[idx_apogee]
    #mask_galah = (idx_galah != -1)
    #mask_ddpayne = (idx_ddpayne != -1)
    #mask_apogee = (idx_apogee != -1)
    #print('{:d} matches.'.format(len(d_galah)))
    #
    ## Save matches
    #fname = 'data/crossmatches_galah_apogee_ddpayne.h5'
    #dset_kw = dict(chunks=True, compression='gzip', compression_opts=3)
    #with h5py.File(fname, 'w') as f:
    #    f.create_dataset('/galah', data=d_galah, **dset_kw)
    #    f.create_dataset('/ddpayne', data=d_ddpayne, **dset_kw)
    #    f.create_dataset('/apogee', data=d_apogee, **dset_kw)
    #    f.create_dataset('/mask_galah', data=mask_galah.astype('u1'), **dset_kw)
    #    f.create_dataset('/mask_ddpayne', data=mask_ddpayne.astype('u1'), **dset_kw)
    #    f.create_dataset('/mask_apogee', data=mask_apogee.astype('u1'), **dset_kw)
    #
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
    #anchor = 'lamost'
    #comparisons = ('apogee', 'galah')
    
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
    
    # Fit offsets and calibrate uncertainties
    from calibrate_errors import get_log_posterior, fit_data
    from scipy.optimize import minimize
    
    param_scale = {'teff': 50, 'logg': 0.05, 'feh': 0.05}
    
    fit_params = {}
    survey_order = ('galah', 'apogee', 'lamost')
    
    for name in param_names:
        print(f'Calibrating uncertainties for {name} ...')
        
        x_obs = np.empty((3,d_galah.size))
        x_err = np.empty((3,d_galah.size))
        
        for i,survey in enumerate(survey_order):
            x_obs[i] = params[name][survey]
            x_err[i] = param_errs[name][survey]
            idx_bad = ~survey_masks[survey]
            x_obs[i][idx_bad] = np.nan
            x_err[i][idx_bad] = np.nan
        
        # Fit (a,b,alpha,beta)
        scale = param_scale[name]
        #theta_guess = np.zeros(8)
        #theta_guess += 0.1 * np.random.normal(size=theta_guess.size)
        #ln_post = get_log_posterior(
        #    x_obs, x_err,
        #    0.1, 0.1*scale,
        #    0.1, 1.0*scale
        #)
        #res = minimize(lambda th: -ln_post(th), theta_guess)

        #a_fit = np.ones(3)
        #b_fit = np.hstack([0., res['x'][:2]])
        ##alpha_fit = 1. + np.exp(res['x'][2:5])
        #alpha_fit = np.exp(res['x'][2:5])
        ##beta_fit = np.exp(res['x'][5:8])
        #beta_fit = np.exp(np.full(3, res['x'][5]))

        #print('Survey order:', survey_order)
        #print(f'a* = {a_fit}')
        #print(f'b* = {b_fit}')
        #print(f'alpha* = {alpha_fit}')
        #print(f'beta* = {beta_fit}')
        fit = fit_data(
            x_obs, x_err,
            0.1*scale, 0.1,
            n_iterations=5,
            chi_cut=30.
        )
        
        fit_params[name] = {}
        for k,s in enumerate(survey_order):
            fit_params[name][s] = {
                'a': fit['a'][k],
                'b': fit['b'][k],
                'alpha': fit['alpha'][k],
                'beta': np.abs(fit['beta'][k])
            }
    
    # Plot histogram of chi, using modified values & uncertainties
    comparisons = [
        ('apogee','lamost'),
        ('galah', 'lamost'),
        ('apogee', 'galah')
    ]
    
    param_split = {
        'teff': 100.,
        'feh': 0.1,
        'logg': 0.1
    }
    
    print('Plotting histograms of chi ...')
    for name in param_names:
        fig,ax_list = plt.subplots(4,3, figsize=(12,10))
        fig.suptitle(rf'$\Delta {param_labels[name]}$')
        
        p = fit_params[name]
        vsplit = param_split[name]
        
        for k,(comp,anchor) in enumerate(comparisons):
            val0 = params[name][anchor]
            var0 = param_errs[name][anchor]**2
            valp0 = p[anchor]['a']*val0 + p[anchor]['b']
            varp0 = (
                p[anchor]['alpha']**2 * var0
                + p[anchor]['b']**2
            )
            
            val1 = params[name][comp]
            var1 = param_errs[name][comp]**2
            
            valp1 = p[comp]['a']*val1 + p[comp]['b']
            varp1 = (
                p[comp]['alpha']**2 * var1
                + p[comp]['b']**2
            )
            
            chi = (valp1-valp0) / np.sqrt(var0+var1)
            chip = (valp1-valp0) / np.sqrt(varp0+varp1)
            
            idx = survey_masks[comp] & survey_masks[anchor]
            idx_lowerr = (np.sqrt(var0+var1) < vsplit)
            chipe0 = chip[idx & idx_lowerr]
            chipe1 = chip[idx & ~idx_lowerr]
            chi = chi[idx]
            chip = chip[idx]
            print(f'{comp} & {anchor}: {np.count_nonzero(idx)} matches.')
            
            print('chi percentiles:')
            pctiles = [0., 1., 10., 50., 90., 99., 100.]
            pct_vals = np.percentile(chi, pctiles)
            pct_valps = np.percentile(chip, pctiles)
            for pct,val,valp in zip(pctiles, pct_vals, pct_valps):
                print(f'  {pct:.0f} : {val:.03f} {valp:.03f}')
            
            ax, axp, axe0, axe1 = ax_list[:,k]
            
            _,x_edges,_ = ax.hist(chi, range=(-5., 5.), bins=50, alpha=0.7)
            _,x_edges,_ = axp.hist(chip, range=(-5., 5.), bins=50, alpha=0.7)
            _,x_edges,_ = axe0.hist(chipe0, range=(-5., 5.), bins=50, alpha=0.7)
            _,x_edges,_ = axe1.hist(chipe1, range=(-5., 5.), bins=50, alpha=0.7)
            print(x_edges)
            
            dx = x_edges[1]-x_edges[0]
            x_gauss = np.linspace(-5., 5., 1000)
            y_gauss = dx * chi.size * np.exp(-0.5*x_gauss**2) / np.sqrt(2.*np.pi)
            ax.plot(x_gauss, y_gauss, c='g', alpha=0.5)
            axp.plot(x_gauss, y_gauss, c='g', alpha=0.5)
            ne0 = chipe0.size / chi.size
            ne1 = chipe1.size / chi.size
            axe0.plot(x_gauss, y_gauss*ne0, c='g', alpha=0.5)
            axe1.plot(x_gauss, y_gauss*ne1, c='g', alpha=0.5)
            
            for a in (ax,axp,axe0,axe1):
                a.grid('on', alpha=0.5)
                a.xaxis.set_major_locator(ticker.AutoLocator())
                a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                a.set_yticklabels([])
            
            ax.set_title(
                rf'${survey_labels[comp]} - {survey_labels[anchor]}\ '
                rf'\left( {chi.size} \right)$'
            )
            ax.set_xticklabels([])
            axp.set_xticklabels([])
            axe0.set_xticklabels([])
            
            if k == 0:
                ax.set_ylabel(r'$\chi$')
                axp.set_ylabel(r'$\chi^{{\prime}}$')
                axe0.set_ylabel(rf'$\chi^{{\prime}},\ \sigma < {vsplit}$')
                axe1.set_ylabel(rf'$\chi^{{\prime}},\ \sigma > {vsplit}$')
            
            fit_label = (
                  rf'$\alpha = {p[comp]["alpha"]:.3f}, {p[anchor]["alpha"]:.3f}$'
                + '\n'
                + rf'$\beta = {p[comp]["beta"]:.3f}, {p[anchor]["beta"]:.3f}$'
            )
            axp.text(
                0.05, 0.95,
                fit_label,
                ha='left', va='top',
                transform=axp.transAxes
            )
        
        fig.subplots_adjust(
            top=0.90, bottom=0.10,
            right=0.95, left=0.05,
            hspace=0.10, wspace=0.08
        )
        
        fig.savefig(os.path.join(fig_dir, f'chip_hist_{name}.{ext}'), dpi=150)
        plt.close(fig)
    
    return 0
    
    # Plot histogram of residuals, as a function of combined uncertainty
    sigma_comb = [np.sqrt(s1**2+s2**2) for s1,s2 in val_errs]
    
    sigma_0 = [50., 0.05, 0.03]
    
    err_floors = [25., 0.07, 0.04]
    err_scales = [0.65, 1., 1.0]
    
    for dval,sigma,sig_0,label,name,sfl,ssc in zip(dvals, sigma_comb, sigma_0,
                                                   labels, names, err_floors,
                                                   err_scales):
        s_edges = [0., sig_0, 2*sig_0, 4*sig_0, 8*sig_0]
        
        n_ax = len(s_edges)-1
        fig,ax_list = plt.subplots(n_ax//2,2, figsize=(8,n_ax+1))
        ax_list.shape = (-1,)
        fig.suptitle(r'$\chi \ (\mathrm{DDPayne}-\mathrm{APOGEE\ DR16}+\mathrm{zp})$')
        
        for k,(s0,s1,ax) in enumerate(zip(s_edges[:-1],s_edges[1:],ax_list)):
            idx = (sigma > s0) & (sigma < s1)
            chi = (dval[idx]-np.median(dval)) / np.sqrt(ssc**2 * sigma[idx]**2 + sfl**2)
            bins = max(min(int(0.1*np.count_nonzero(idx)), 100), 10)
            _,x_edges,_ = ax.hist(chi, range=(-5.,5.), bins=bins, alpha=0.7)
            dx = x_edges[1]-x_edges[0]
            x = np.linspace(-5., 5., 1000)
            y_gauss = dx * chi.size * np.exp(-0.5*x**2) / np.sqrt(2.*np.pi)
            ax.plot(x, y_gauss, c='g', alpha=0.5)
            ax.grid('on', alpha=0.5)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.text(
                0.05, 0.95,
                r'${:.3g} < \sigma < {:.3g}$'.format(s0,s1),
                ha='left',
                va='top',
                transform=ax.transAxes
            )
            if (k >= n_ax-2):
                print(k)
                ax.set_xlabel(r'$'+label+'$')
            else:
                ax.set_xticklabels([])
        
        fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.05)
        
        fig.savefig(
            '/n/fink2/www/ggreen/dd_stellar_models/ddpayne_vs_apogee_chi_hist_vs_err_'+name+ext,
            dpi=150
        )
        plt.close(fig)
    
    # Plot histogram of scores
    fig,ax_list = plt.subplots(3,1, figsize=(6,12))
    fig.suptitle(r'$\chi \ (\mathrm{DDPayne}-\mathrm{APOGEE\ DR16}+\mathrm{zp})$')
    
    chis = [
        (dval-np.median(dval)) / np.sqrt(s1**2+s2**2)
        for dval,(s1,s2) in zip(dvals, val_errs)
    ]
    
    for ax,chi,label in zip(ax_list, chis, labels):
        _,x_edges,_ = ax.hist(chi, range=(-5.,5.), bins=100, alpha=0.7)
        dx = x_edges[1]-x_edges[0]
        x = np.linspace(-5., 5., 1000)
        y_gauss = dx * chi.size * np.exp(-0.5*x**2) / np.sqrt(2.*np.pi)
        ax.plot(x, y_gauss, c='g', alpha=0.5)
        ax.grid('on', alpha=0.5)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_xlabel(r'$'+label+'$')
    
    fig.subplots_adjust(top=0.95, bottom=0.05)
    
    fig.savefig('/n/fink2/www/ggreen/dd_stellar_models/ddpayne_vs_apogee_chi_hist'+ext)
    plt.close(fig)
    
    # Plot histogram of scores, using error floors
    chi_primes = [
        (dval-np.median(dval)) / np.sqrt(sc**2 * (e1**2 + e2**2) + 2*fl**2)
        for dval,(e1,e2),sc,fl in zip(dvals, val_errs, err_scales, err_floors)
    ]
    
    for log in [False, True]:
        fig,ax_list = plt.subplots(3,1, figsize=(6,12))
        fig.suptitle(r'$\chi^{\prime} \ (\mathrm{DDPayne}-\mathrm{APOGEE\ DR16}+\mathrm{zp})$')
        
        for ax,chi_p,label in zip(ax_list, chi_primes, labels):
            _,x_edges,_ = ax.hist(chi_p, range=(-5.,5.), bins=100, alpha=0.7, log=log)
            dx = x_edges[1]-x_edges[0]
            x = np.linspace(-5., 5., 1000)
            y_gauss = dx * chi_p.size * np.exp(-0.5*x**2) / np.sqrt(2.*np.pi)
            ax.plot(x, y_gauss, c='g', alpha=0.5)
            ax.grid('on', alpha=0.5)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.set_xlabel(r'$'+label+'$')
        
        fig.subplots_adjust(top=0.95, bottom=0.05)
        
        fname = '/n/fink2/www/ggreen/dd_stellar_models/ddpayne_vs_apogee_chip_hist'
        if log:
            fname += '_log'
        fig.savefig(fname+ext)
        plt.close(fig)
    
    # Plot correlations of residuals
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(r'$\chi \ (\mathrm{DDPayne}-\mathrm{APOGEE\ DR16}+\mathrm{zp})$')
    
    for row,dy in enumerate(chis[1:]):
        for col,dx in enumerate(chis[:row+1]):
            ax = fig.add_subplot(2,2,2*row+col+1)
            #ax.scatter(
            #    dx, dy,
            #    s=3, edgecolors='none',
            #    c='b', alpha=0.1,
            #    rasterized=True
            #)
            ax.hexbin(
                dx, dy,
                extent=(-5.,5.,-5.,5.),
                gridsize=50,
                linewidths=0.25
            )
            ax.grid('on', alpha=0.5)
            ax.set_xlim(-5., 5.)
            ax.set_ylim(-5., 5.)
            
            if row == 1:
                ax.set_xlabel(r'$'+labels[col]+r'$')
            else:
                ax.set_xticklabels([])
            
            if col == 0:
                ax.set_ylabel(r'$'+labels[row+1]+r'$')
            else:
                ax.set_yticklabels([])
    
    fig.savefig('/n/fink2/www/ggreen/dd_stellar_models/ddpayne_vs_apogee_corrs'+ext)
    plt.close(fig)
    
    # Scatterplots
    fig,ax_list = plt.subplots(1,3, figsize=(11,4))
    for ax,(val1,val2),label in zip(ax_list, vals, labels):
        ax.scatter(
            val1, val2,
            s=3, edgecolors='none',
            c='b', alpha=0.1,
            rasterized=True
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = max(xlim[0], ylim[0])
        x1 = min(xlim[1], ylim[1])
        w = x1 - x0
        x0 -= 0.25*w
        x1 += 0.25*w
        xlim = (x0, x1)
        ax.plot(xlim, xlim, c='k', lw=1., alpha=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel(r'$\mathrm{DDPayne}$')
        ax.set_title(r'$'+label+r'$')
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_list[0].set_ylabel(r'$\mathrm{APOGEE}$')
    fig.subplots_adjust(wspace=0.2, left=0.08, right=0.98, bottom=0.12, top=0.90)
    
    fig.savefig('/n/fink2/www/ggreen/dd_stellar_models/ddpayne_vs_apogee_scatter'+ext)
    plt.close(fig)
    
    ## Plot dT in (logg,T) plane
    #teff = d_ddpayne['ddpayne_teff']
    #logg = d_ddpayne['ddpayne_logg']
    #feh = d_ddpayne['ddpayne_feh']
    
    return 0


if __name__ == '__main__':
    main()
