#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division


import numpy as np
import h5py
import os

import lsd


np.seterr(invalid='ignore', divide='ignore')


def filter_galaxies(d):
    """
    Returns a copy of the dataset with objects determined by any
    survey to be extended/galaxies/binaries removed.
    """
    idx_bad = (
          np.any(
              (d['ps1_mag'] - d['ps1_apmag'] > 0.1)
            & np.isfinite(d['ps1_mag'])
            & np.isfinite(d['ps1_apmag']),
            axis=1
          )
        | (d['tmass_ext_key'] > 0)
    )
    d = d[~idx_bad]
    return d


def filter_ps1(d):
    idx_good = (d['ps1.nmag_ok'] > 0)
    d['ps1_mag'][~idx_good] = np.nan
    d['ps1_mag_err'][~idx_good] = np.nan


def filter_tmass(d):
    '''
    Set observed 2MASS magnitudes that fail high-quality cut
    (as per the 2MASS recommendations) to NaN. See
    <http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_6b.html#composite>
    '''
    
    # Photometric quality in each passband
    idx = (d['tmass_ph_qual'] == '0')
    d['tmass_ph_qual'][idx] = '000'
    ph_qual = np.array(map(list, d['tmass_ph_qual']))
    
    # Read flag in each passband
    idx = (d['tmass_rd_flg'] == '0')
    d['tmass_rd_flg'][idx] = '000'
    rd_flg = np.array(map(list, d['tmass_rd_flg']))
    
    # Contamination flag in each passband
    idx = (d['tmass_cc_flg'] == '0')
    d['tmass_cc_flg'][idx] = '000'
    cc_flg = np.array(map(list, d['tmass_cc_flg']))
    
    # Combine passband flags
    cond_1 = (ph_qual == 'A') | (rd_flg == '1') | (rd_flg == '3')
    cond_1 &= (cc_flg == '0')
    
    # Source quality flags
    cond_2 = (d['tmass_use_src'] == 1) & (d['tmass_gal_contam'] == 0)
    
    # Set mags that fail quality cuts to NaN
    for i,b in enumerate('JHK'):
        idx = ~(cond_1[:,i] & cond_2)
        d['tmass_{}_mag'.format(b)][idx] = np.nan
        d['tmass_{}_mag_err'.format(b)][idx] = np.nan


def filter_unwise(d):
    # Filter out bands with bad flags or high chi^2/d.o.f.
    idx = (d['unwise_flags'] != 0) | (d['unwise_rchi2'] > 5.)
    d['unwise_mag'][idx] = np.nan
    d['unwise_mag_err'][idx] = np.nan
    
    # Convert unWISE from Vega to AB
    d['unwise_mag'] += np.array([2.699, 3.339])[None,:]


def filter_galah(d, spec_snr_min=20):
    idx_good = (
          (d['galah_flag_cannon'] == 0)
        & (
              (d['galah_snr_c1'] > spec_snr_min)
            | (d['galah_snr_c2'] > spec_snr_min)
            | (d['galah_snr_c3'] > spec_snr_min)
            | (d['galah_snr_c4'] > spec_snr_min)
          ) 
        & (
              (d['galah_teff_err'] > 1.e-5)
            & (d['galah_logg_err'] > 1.e-5)
            & (d['galah_feh_err'] > 1.e-5)
          )
        & (
              (d['galah_teff'] > 1.)
            & (d['galah_feh'] > -9.)
            & (d['galah_logg'] > -2.)
          )
    )
    n_bad = np.count_nonzero(~idx_good)
    print(r'GALAH: {:d} ({:.3f}%) bad'.format(n_bad, (100.*n_bad)/idx_good.size))
    for p in ('teff', 'logg', 'feh'):
        d['galah_{}'.format(p)][~idx_good] = np.nan
        d['galah_{}_err'.format(p)][~idx_good] = np.nan


def filter_sdss(d):
    idx_good = (
          ~(d['sdss_aspcap_flag'] & 128).astype('bool')     # STAR_WARN (bit 7)
        & ~(d['sdss_aspcap_flag'] & 8388608).astype('bool') # STAR_BAD (bit 23)
        & (
              (d['sdss_aspcap_teff_err'][:] > 1.e-5) # T_eff
            & (d['sdss_aspcap_logg_err'][:] > 1.e-5) # log(g)
            & (d['sdss_aspcap_m_h_err'][:] > 1.e-5) # [M/H]
          )
        & (
              (d['sdss_aspcap_param'][:,0] > 1.)   # T_eff
            & (d['sdss_aspcap_param'][:,1] > -10.) # log(g)
            & (d['sdss_aspcap_param'][:,3] > -5.)  # [M/H]
          )
    )
    n_bad = np.count_nonzero(~idx_good)
    print(r'SDSS: {:d} of {:d} bad ({:.3f}%)'.format(
        n_bad,
        idx_good.size,
        (100.*n_bad)/idx_good.size
    ))
    d['sdss_aspcap_param'][~idx_good] = np.nan
    d['sdss_aspcap_fparam_cov'][~idx_good] = np.nan


def filter_ddpayne(d, spec_snr_min=20):
    idx_good = (
          (
              (d['ddpayne_snr_u'] > spec_snr_min)
            | (d['ddpayne_snr_g'] > spec_snr_min)
            | (d['ddpayne_snr_r'] > spec_snr_min)
            | (d['ddpayne_snr_i'] > spec_snr_min)
            | (d['ddpayne_snr_z'] > spec_snr_min)
          ) 
        & (d['ddpayne_qflag_chi2'] == b'good')
        & (d['ddpayne_flag_singlestar'] == b'YES')
        & (
              (d['ddpayne_teff_err'] > 1.e-5)
            & (d['ddpayne_logg_err'] > 1.e-5)
            & (d['ddpayne_feh_err'] > 1.e-5)
          )
        & (
              (d['ddpayne_teff'] > 1.)
            & (d['ddpayne_feh'] > -9.)
            & (d['ddpayne_logg'] > -2.)
          )
    )
    for p in ('teff', 'logg', 'feh'):
        d['ddpayne_{}'.format(p)][~idx_good] = np.nan
        d['ddpayne_{}_err'.format(p)][~idx_good] = np.nan


def require_spec_params(d):
    idx_good = (
          np.isfinite(d['sdss_aspcap_param'][:,0])
        & np.isfinite(d['sdss_aspcap_param'][:,1])
        & np.isfinite(d['sdss_aspcap_param'][:,3])
    )
    d = d[idx_good]
    return d


def filter_data(d):
    filter_sdss(d)
    filter_ps1(d)
    filter_tmass(d)
    filter_unwise(d)
    d = filter_galaxies(d)
    d = require_spec_params(d)
    return d


def main():
    dset_name = 'stellar_phot_spec_ast'
    
    for l0 in np.arange(0., 350.01, 10.):
        print('l in ({:.0f}, {:.0f})'.format(l0,l0+10))
        
        lon_bounds = (l0, l0+10.-1.e-8)#(60., 70.)#359.99999)
        lat_bounds = (-90., 90.)
        #lon_bounds = (42., 48.)
        #lat_bounds = (42., 48.)
        query_bounds = []
        query_bounds.append(
            lsd.bounds.rectangle(
                lon_bounds[0], lat_bounds[0],
                lon_bounds[1], lat_bounds[1],
                coordsys='gal'
            )
        )
        query_bounds = lsd.bounds.make_canonical(query_bounds)
        
        db = lsd.DB(os.environ['LSD_DB'])
        
        query = (
            "SELECT "
            ""   # Gaia
            "    gaia.ra as ra, gaia.dec as dec, "
            "    gaia.l as gal_l, gaia.b as gal_b, "
            "    gaia.parallax as parallax, "
            "    gaia.parallax_over_error as parallax_over_err, "
            "    gaia.parallax_error as parallax_err, "
            "    gaia.astrometric_chi2_al as ast_chi2, "
            "    gaia.astrometric_n_good_obs_al as ast_n_good_obs, "
            "    gaia.visibility_periods_used as visibility_periods_used, "
            "    gaia.phot_g_mean_mag as gaia_g_mag, "
            "    2.5/log(10.)/gaia.phot_g_mean_flux_over_error as gaia_g_mag_err, "
            "    gaia.phot_bp_mean_mag as gaia_bp_mag, "
            "    2.5/log(10.)/gaia.phot_bp_mean_flux_over_error as gaia_bp_mag_err, "
            "    gaia.phot_rp_mean_mag as gaia_rp_mag, "
            "    2.5/log(10.)/gaia.phot_rp_mean_flux_over_error as gaia_rp_mag_err, "
            "    gaia.phot_g_n_obs as gaia_g_n_obs, "
            "    gaia.phot_bp_n_obs as gaia_bp_n_obs, "
            "    gaia.phot_rp_n_obs as gaia_rp_n_obs, "
            ""   # SFD
            "    SFD.EBV(gal_l, gal_b) as SFD, "
            ""   # SDSS
            "    sdss.survey as sdss_survey, "
            "    sdss.starflag as sdss_starflag, "
            "    sdss.starflags as sdss_starflag_verbose, "
            "    sdss.param as sdss_aspcap_param, "
            "    sdss.fparam_cov as sdss_aspcap_fparam_cov, "
            "    sdss.teff_err as sdss_aspcap_teff_err, "
            "    sdss.logg_err as sdss_aspcap_logg_err, "
            "    sdss.m_h_err as sdss_aspcap_m_h_err, "
            "    sdss.alpha_m_err as sdss_aspcap_alpha_m_err, "
            "    sdss.snr as sdss_snr, "
            "    sdss.aspcapflag as sdss_aspcap_flag, "
            "    sdss.aspcapflags as sdss_aspcap_flag_verbose, "
            "    sdss.aspcap_chi2 as sdss_aspcap_chi2, "
            "    sdss.gaia_r_est as gaia_r_est, "
            "    sdss.gaia_r_lo as gaia_r_lo, "
            "    sdss.gaia_r_hi as gaia_r_hi, "
            ""   # PS1
            "    2.5/log(10.)*ps1.err/ps1.mean as ps1_mag_err, "
            "    -2.5*log10(ps1.mean) as ps1_mag, "
            "    -2.5*log10(mean_ap) as ps1_apmag, "
            "    ps1.nmag_ok, "
            ""   # 2MASS
            "    tmass.ph_qual as tmass_ph_qual, "
            "    tmass.use_src as tmass_use_src, "
            "    tmass.rd_flg as tmass_rd_flg, "
            "    tmass.ext_key as tmass_ext_key, "
            "    tmass.gal_contam as tmass_gal_contam, "
            "    tmass.cc_flg as tmass_cc_flg, "
            "    tmass.j_m as tmass_J_mag, "
            "    tmass.j_msigcom as tmass_J_mag_err, "
            "    tmass.h_m as tmass_H_mag, "
            "    tmass.h_msigcom as tmass_H_mag_err, "
            "    tmass.k_m as tmass_K_mag, "
            "    tmass.k_msigcom as tmass_K_mag_err, "
            ""   # UNWISE
            "    22.5-2.5*log10(unwise.flux) as unwise_mag, "
            "    2.5/log(10.)*unwise.dflux/unwise.flux as unwise_mag_err, "
            "    unwise.flags_unwise as unwise_flags, "
            "    unwise.flags_info as unwise_flags_info, "
            "    unwise.rchi2 as unwise_rchi2 "
            "FROM "
            "    gaia_dr2_source as gaia, "
            "    sdss_dr16_allstar(inner, matchedto=gaia, dmax=0.2, nmax=1) as sdss, "
            "    ucal_fluxqz(outer, matchedto=gaia, dmax=0.2, nmax=1) as ps1, "
            "    tmass(outer, matchedto=gaia, dmax=0.4, nmax=1), "
            "    unwise_obj_primary(outer, matchedto=gaia, dmax=0.4, nmax=1) as unwise "
            "WHERE "
            "    (visibility_periods_used > 8) "
            "    & (ast_chi2 / (ast_n_good_obs - 5.) < 1.44 * np.clip(np.exp(-0.4 * (gaia_g_mag-19.5)), 1., np.inf)) "
            "    & (gaia_g_mag_err < {phot_err_max}) "
            "    & (gaia_g_n_obs > 2) "
            "    & (SFD < {SFD_max}) "
        ).format(
            phot_err_max=0.2,
            SFD_max=5.0
        )
        
        rows = db.query(query).fetch(bounds=query_bounds, nworkers=3)
        rows = rows.as_ndarray()
        print('{:d} sources.'.format(len(rows)))
        rows = filter_data(rows)
        print('{:d} sources after filtering.'.format(len(rows)))
        print(rows.dtype)
        print(rows[:10])
        for key in rows.dtype.names:
            print('')
            print(key)
            print(rows[key][:10])
        
        fname = 'data/dr16_data_{:.0f}to{:.0f}.h5'.format(l0,l0+10)
        with h5py.File(fname, 'w') as f:
            f.create_dataset(
                '/'+dset_name,
                data=rows,
                chunks=True,
                compression='gzip',
                compression_opts=3
            )
    
    return 0


if __name__ == '__main__':
    main()

