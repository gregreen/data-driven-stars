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
        | (d['ddpayne_flag_singlestar'] != b'YES')
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
            & (d['ddpayne_feh'] > -10.)
            & (d['ddpayne_logg'] > -5.)
          )
    )
    for p in ('teff', 'logg', 'feh'):
        d['ddpayne_{}'.format(p)][~idx_good] = np.nan
        d['ddpayne_{}_err'.format(p)][~idx_good] = np.nan


def require_spec_params(d):
    idx_good = (
          np.isfinite(d['ddpayne_teff'])
        & np.isfinite(d['ddpayne_logg'])
        & np.isfinite(d['ddpayne_feh'])
    )
    d = d[idx_good]
    return d


def filter_data(d):
    #filter_ddpayne(d)
    filter_ps1(d)
    filter_tmass(d)
    filter_unwise(d)
    #d = filter_galaxies(d)
    #d = require_spec_params(d)
    return d


def main():
    dset_name = 'stellar_phot_spec_ast'
    
    for l0 in np.arange(0., 350.01, 10.):
        print('l in ({:.0f}, {:.0f})'.format(l0,l0+10))
        
        lon_bounds = (l0, l0+10.-1.e-8)#(60., 70.)#359.99999)
        lat_bounds = (-90., 90.)
        #lon_bounds = (60., 90.)#359.99999)
        #lat_bounds = (-90., 90.)
        #lon_bounds = (55., 60.)
        #lat_bounds = (45., 48.)
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
            "    gaia.source_id as gdr2_source_id, "
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
            "    gaia.phot_bp_rp_excess_factor as gaia_bp_rp_excess, "
            ""   # SFD
            "    SFD.EBV(gal_l, gal_b) as SFD, "
            ""   # LAMOST DDPAYNE
            "    ddpayne.teff as ddpayne_teff, "
            "    ddpayne.teff_err as ddpayne_teff_err, "
            "    ddpayne.teff_flag as ddpayne_teff_flag, "
            "    ddpayne.logg as ddpayne_logg, "
            "    ddpayne.logg_err as ddpayne_logg_err, "
            "    ddpayne.logg_flag as ddpayne_logg_flag, "
            "    ddpayne.feh as ddpayne_feh, "
            "    ddpayne.feh_err as ddpayne_feh_err, "
            "    ddpayne.feh_flag as ddpayne_feh_flag, "
            "    ddpayne.snr_u as ddpayne_snr_u, "
            "    ddpayne.snr_g as ddpayne_snr_g, "
            "    ddpayne.snr_r as ddpayne_snr_r, "
            "    ddpayne.snr_i as ddpayne_snr_i, "
            "    ddpayne.snr_z as ddpayne_snr_z, "
            "    ddpayne.chi2 as ddpayne_chi2, "
            "    ddpayne.chi2ratio as ddpayne_chi2ratio, "
            "    ddpayne.qflag_chi2 as ddpayne_qflag_chi2, "
            "    ddpayne.flag_singlestar as ddpayne_flag_singlestar, "
            "    ddpayne.subclass as ddpayne_subclass, "
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
            "    lamost_dr5_ddpayne(inner, matchedto=gaia, dmax=0.2, nmax=1) as ddpayne, "
            "    ucal_fluxqz(outer, matchedto=gaia, dmax=0.2, nmax=1) as ps1, "
            "    tmass(outer, matchedto=gaia, dmax=0.4, nmax=1), "
            "    unwise_obj_primary(outer, matchedto=gaia, dmax=0.4, nmax=1) as unwise "
            "WHERE "
            # Gaia quality flags
            "    (visibility_periods_used > 8) "
            "    & (ast_chi2 / (ast_n_good_obs - 5.) < 1.44 * np.clip(np.exp(-0.4 * (gaia_g_mag-19.5)), 1., np.inf)) "
            "    & (gaia_g_mag_err < {phot_err_max}) "
            "    & (gaia_g_n_obs > 2) "
            #"    & ((1.+0.015*(gaia_bp_mag-gaia_rp_mag)**2) < gaia_bp_rp_excess) "
            #"    & ((1.3+0.06*(gaia_bp_mag-gaia_rp_mag)**2) > gaia_bp_rp_excess) "
            # SFD cut
            "    & (SFD < {SFD_max}) "
            # Cut out galaxies
            "    & ~( "
            "           np.any( "
            "              (ps1_mag - ps1_apmag > 0.1) "
            "            & np.isfinite(ps1_mag) "
            "            & np.isfinite(ps1_apmag), "
            "            axis=1 "
            "           ) "
            "         | (tmass_ext_key > 0) "
            "      ) "
            # DDPayne quality cuts
            "    & ( "
            "          (ddpayne_snr_u > spec_snr_min) "
            "        | (ddpayne_snr_g > spec_snr_min) "
            "        | (ddpayne_snr_r > spec_snr_min) "
            "        | (ddpayne_snr_i > spec_snr_min) "
            "        | (ddpayne_snr_z > spec_snr_min) "
            "      ) "
            "    & (ddpayne_qflag_chi2 == b'good') "
            "    & (ddpayne_flag_singlestar == b'YES') "
            "    & ( "
            "          (ddpayne_teff_err > 1.e-5) "
            "        & (ddpayne_logg_err > 1.e-5) "
            "        & (ddpayne_feh_err > 1.e-5) "
            "      ) "
            "    & ( "
            "          (ddpayne_teff > 1.) "
            "        & (ddpayne_feh > -10.) "
            "        & (ddpayne_logg > -5.) "
            "      ) "
        ).format(
            spec_snr_min=20.,
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
        
        fname = 'data/ddpayne_data_{:.0f}to{:.0f}.h5'.format(l0,l0+10)
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

