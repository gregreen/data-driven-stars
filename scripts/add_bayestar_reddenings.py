#!/usr/bin/env python

from __future__ import print_function, division


import numpy as np
from scipy.stats import truncnorm
import h5py

from glob import glob

from dustmaps.bayestar import BayestarQuery
from dustmaps.sfd import SFDQuery


def load_data(fname):
    with h5py.File(fname, 'r') as f:
        d = f['stellar_phot_spec_ast'][:]
    return d


def query_reddening(q_b19, d, n_samples=25):
    """
    Queries the reddening and its uncertainty for each star, taking into
    account that Bayestar19 has uncertainties at each distance, and that
    stellar distances are uncertain.
    """

    # Identify stars without parallaxes
    idx = ~np.isfinite(d['parallax']) | ~np.isfinite(d['parallax_err'])
    d['parallax'][idx] = 1.
    d['parallax_err'][idx] = 0.1

    # Draw parallaxes from Gaia plx likelihood for each star (clipped to >0).
    # This is equivalent to putting a flat prior on parallax (for plx > 0).
    # Could also use another choice of prior, such as Coryn-Bailer Jones'
    # distances.
    plx = np.empty((d.size, n_samples), dtype='f4')
    for k,(mu,sigma) in enumerate(zip(d['parallax'], d['parallax_err'])):
        #if ~idx[k]:
        #    plx[k,:] = 1.
        #    continue
        try:
            plx[k,:] = truncnorm.rvs(0., np.inf, mu, sigma, size=n_samples)
        except Exception as e:
            print(mu, sigma)
            raise(e)

    # Replace negative parallaxes (truncnorm sometimes fails)
    plx[plx < 0.] = 1.e-5

    # Naive transformation of parallax into distance
    r = 1. / plx # in kpc
    
    # Sample reddenings, taking into account uncertainty in distance
    # and Bayestar19 estimates
    E = np.empty((d.size, n_samples), dtype='f4')
    for k in range(n_samples):
        E[:,k] = q_b19.query_gal(
            d['gal_l'],
            d['gal_b'],
            d=r[:,k],
            mode='random_sample'
        )
    
    # Store mean and std. dev. of reddening samples
    E0 = np.mean(E, axis=1)
    sigma_E = np.std(E, axis=1)

    # Stars without parallaxes get NaN reddenings
    E0[idx] = np.nan
    sigma_E[idx] = np.nan

    return E0, sigma_E


def save_bayestar(fname, E0, sigma_E):
    """
    Saves reddening and its uncertainty to the stellar data file.
    """
    with h5py.File(fname, 'a') as f:
        if 'reddening' in f:
            f['reddening'][...] = E0
        else:
            f.create_dataset(
                'reddening',
                data=E0,
                chunks=True,
                compression='gzip',
                compression_opts=3
            )

        if 'reddening_err' in f:
            f['reddening_err'][...] = sigma_E
        else:
            f.create_dataset(
                'reddening_err',
                data=sigma_E,
                chunks=True,
                compression='gzip',
                compression_opts=3
            )


def main():
    q_b19 = BayestarQuery(version='bayestar2019')#, max_samples=4)

    fnames = (
          glob('data/dr16_data_*to*.h5')
        + glob('data/ddpayne_data_*to*.h5')
        + glob('data/galah_data_*to*.h5')
    )
    print(f'{len(fnames):d} files.')
    for fn in fnames:
        print(f'Adding reddenings to {fn} ...')
        d = load_data(fn)
        E0, sigma_E = query_reddening(q_b19, d)
        save_bayestar(fn, E0, sigma_E)

    return 0

if __name__ == '__main__':
    main()

