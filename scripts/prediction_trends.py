#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py
import scipy.stats

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from plot_utils import correlation_plot


fmt = 'svg'


def load_predictions(fname):
    """
    Loads neural network predictions:
        data       - Input data
        y_obs      - Observed (G, X1-G, X2-G, ...)
        y_pred     - Predicted (G, X1-G, X2-G, ...)
        cov_y      - Covariance for each observation
        reddening  - Inferred reddening of each star
        R          - Inferred reddening vector
    Returns a dictionary with each of the above elements.
    """
    d = {}

    # Load all fields stored in file
    with h5py.File(fname, 'r') as f:
        for key in f.keys():
            d[key] = f[key][:]
        d['R'] = f.attrs['R'][:]

    # Calculate reddened predictions
    d['y_pred_red'] = d['y_pred'] + d['reddening'][:,None]*d['R'][None,:]

    # Calculate chi^2/d.o.f.
    dy = d['y_pred_red'] - d['y_obs']

    icov = np.empty_like(d['cov_y'])
    for k,c in enumerate(d['cov_y']):
        icov[k] = np.linalg.inv(c)

    #nu = dy.shape[1] # Number of parameters << number of observations
    #print(f'nu = {nu}')
    obs = np.empty(dy.shape, dtype='bool')
    for k in range(dy.shape[1]):
        obs[:,k] = (d['cov_y'][:,k,k] < 100.)
    d['nu'] = np.count_nonzero(obs, axis=1)
    print(d['nu'])
    print(np.median(d['nu']))
    print(np.mean(d['nu']))

    d['rchi2'] = np.einsum('ni,nij,nj->n', dy, icov, dy) / d['nu']

    return d


def rchi2_hist(pred):
    fig = plt.figure(figsize=(8,6), dpi=150)
    ax = fig.add_subplot(1,1,1)

    bins = np.arange(0., 5.01, 0.1)
    ax.hist(pred['rchi2'], bins=bins, density=True)

    ylim = ax.get_ylim()

    r = np.linspace(0., bins[-1], 1000)
    nu = pred['y_obs'].shape[1]

    for nu_k in range(1,nu+1):
        pdf = scipy.stats.chi2(nu_k).pdf(r*nu_k)*nu_k#/(bins[1]-bins[0])
        ax.plot(r, pdf, c='b', alpha=0.25)

    ax.set_xlim(0., bins[-1])
    ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.savefig(f'plots/rchi2_hist.{fmt}', dpi=150)
    plt.close(fig)


def rchi2_trends(pred):
    y = pred['rchi2']
    y_bins = np.arange(0., 15.01, 0.25)

    x_fields = [
        ('teff', pred['data']['teff'], r'$T_{\mathrm{eff}}$', (3500.,8500.)),
        ('logg', pred['data']['logg'], r'$\log \left( g \right)$', (0.6,5.4)),
        ('feh', pred['data']['feh'], r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', (-2.5,0.5)),
        ('reddening', pred['reddening'], r'$\mathrm{reddening}$', (0.,1.25))
    ]

    for key,x,label,lim in x_fields:
        fig = plt.figure(figsize=(8,6), dpi=150)
        ax = fig.add_subplot(1,1,1)
        
        idx = np.isfinite(x) & np.isfinite(y) & (x > lim[0]) & (x < lim[1])
        correlation_plot(ax, x[idx], y[idx], y_bins=y_bins)#, norm='max')
        
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\chi^2 / \nu$')
        
        fig.savefig(f'plots/rchi2_trend_{key}.{fmt}')
        plt.close(fig)


def residual_trends(pred, comp_with_G=False):
    # Calculate residuals in (G, G-BP, BP-RP, RP-g, g-r, ...)-space,
    # or in (G, BP-G, RP-G, g-G, r-G, ...)-space, depending on whether
    # comp_with_G is False or True.
    dy = pred['y_obs'] - pred['y_pred_red']
    if not comp_with_G:
        dy[:,1:] = dy[:,:-1] - dy[:,1:]

    obs = np.empty(dy.shape, dtype='bool')
    obs[:,0] = (pred['cov_y'][:,0,0] < 100.)
    dy_over_err = np.empty_like(dy)
    dy_over_err[:,0] = dy[:,0] / np.sqrt(pred['cov_y'][:,0,0])
    for k in range(1,dy.shape[1]):
        if comp_with_G:
            dy_over_err[:,k] = dy[:,k] / np.sqrt(pred['cov_y'][:,k,k])
        else:
            c = (
                  pred['cov_y'][:,k,k]
                + pred['cov_y'][:,k-1,k-1]
                - 2. * pred['cov_y'][:,k,k-1]
            )
            #c -= 2. * 0.02**2
            obs[:,k] = (
                  (pred['cov_y'][:,k,k] < 100.)
                & (pred['cov_y'][:,k-1,k-1] < 100.)
            )
            dy_over_err[:,k] = dy[:,k] / np.sqrt(c)

    x_fields = [
        ('teff', pred['data']['teff'], r'$T_{\mathrm{eff}}$', (3500.,8500.)),
        ('logg', pred['data']['logg'], r'$\log \left( g \right)$', (0.6,5.4)),
        ('feh', pred['data']['feh'], r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', (-2.5,0.5)),
        ('reddening', pred['reddening'], r'$\mathrm{reddening}$', (0.,1.25))
    ]

    y_bins = np.arange(-5., 5.01, 0.2)

    bands = ['G', 'BP', 'RP'] + list('grizyJH') + ['K_s','W_1','W_2']
    if comp_with_G:
        y_cols = (
              [bands[0]]
            + [f'{b}-G' for b in bands[1:]]
        )
    else:
        y_cols = (
              [bands[0]]
            + [f'{b1}-{b2}' for b1,b2 in zip(bands[:-1],bands[1:])]
        )

    for k,col_label in enumerate(y_cols):
        col_label_simple = col_label.replace('-','').replace('_','')
        if k != 0:
            col_label = r'\left( {} \right)'.format(col_label)

        y = dy_over_err[:,k]

        for key,x,xlabel,lim in x_fields:
            fig = plt.figure(figsize=(8,6), dpi=150)
            ax = fig.add_subplot(1,1,1)
            
            idx = (
                  np.isfinite(x) & np.isfinite(y) & obs[:,k]
                & (x > lim[0]) & (x < lim[1])
            )
            correlation_plot(ax, x[idx], y[idx], y_bins=y_bins)
            
            ax.set_xlabel(xlabel)

            ylabel = ' - '.join([
                r'{}_{{ \mathrm{{ {} }} }}'.format(col_label,sub)
                for sub in ('obs', 'pred')
            ])
            ylabel = r'$\left[ {} \right] / \sigma$'.format(ylabel)
            ax.set_ylabel(ylabel)
            
            fig.savefig(f'plots/d{col_label_simple}_trend_{key}.{fmt}')
            plt.close(fig)


def main():
    fname = 'data/predictions_rchisqfilt_2hidden_it6.h5'
    pred = load_predictions(fname)

    print('chi^2/nu = {}'.format(np.mean(pred['rchi2'])))

    rchi2_hist(pred)
    rchi2_trends(pred)
    residual_trends(pred, comp_with_G=False)
    residual_trends(pred, comp_with_G=True)

    return 0

if __name__ == '__main__':
    main()

