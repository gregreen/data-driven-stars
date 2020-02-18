#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import scipy.stats
import h5py

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from glob import glob
from time import time
import json


def load_data(fname):
    print(f'Loading {fname} ...')
    with h5py.File(fname, 'r') as f:
        d = f['io_data'][:]
    return d


def get_corr_matrix(cov):
    rho = cov.copy()
    sqrt_cov_diag = np.sqrt(cov[np.diag_indices(cov.shape[0])])
    rho /= sqrt_cov_diag[:,None]
    rho /= sqrt_cov_diag[None,:]
    rho[np.diag_indices(cov.shape[0])] = sqrt_cov_diag
    return rho


def get_inputs_outputs(d, pretrained_model=None, rchisq_max=None):
    n_bands = 13 # Gaia (G, BP, RP), PS1 (grizy), 2MASS (JHK), unWISE (W1,W2)
    n_atm_params = 3 # (T_eff, logg, [M/H])

    # Stellar spectroscopic parameters
    print('Fill in stellar atmospheric parameters ...')
    x = np.empty((d.size,3), dtype='f4')
    x[:] = d['atm_param'][:]

    x_p = np.empty((d.size,3), dtype='f4')
    x_p = d['atm_param_p'][:]

    # Magnitudes
    print('Fill in stellar magnitudes ...')
    y = np.empty((d.size,n_bands), dtype='f4')
    y[:] = d['mag'][:]

    # Covariance of y
    print('Empty covariance matrix ...')
    cov_y = np.zeros((d.size,n_bands,n_bands), dtype='f4')

    # \delta m
    print('Covariance: \delta m ...')
    for i in range(n_bands):
        cov_y[:,i,i] = d['mag_err'][:,i]**2

    # Replace NaN magnitudes with median (in each band).
    # Also set corresponding variances to large number.
    print('Replace NaN magnitudes ...')
    for b in range(n_bands):
        idx = (
              ~np.isfinite(y[:,b])
            | ~np.isfinite(cov_y[:,b,b])
        )
        n_bad = np.count_nonzero(idx)
        n_tot = idx.size
        y0 = np.median(y[~idx,b])
        if np.isnan(y0):
            y0 = 0.
        print(f'Band {b}: {n_bad} of {n_tot} bad. Replacing with {y0:.5f}.')
        y[idx,b] = y0
        cov_y[idx,b,b] = 99.**2.

    # Transform both y and its covariance
    B = np.identity(n_bands, dtype='f4')
    B[1:,0] = -1.

    print('Transform y -> B y ...')
    y = np.einsum('ij,nj->ni', B, y) # y' = B y
    print('Transform C -> B C B^T ...')
    #cov_y = np.einsum('ik,nkl,jl->nij', B, cov_y, B) # C' = B C B^T
    cov_y = np.einsum('nik,jk->nij', cov_y, B)
    cov_y = np.einsum('ik,nkj->nij', B, cov_y)

    # Add in dM/dtheta term
    if pretrained_model is not None:
        print('Calculate J = dM/dtheta ...')
        J = calc_dmag_color_dtheta(pretrained_model, x_p)
        cov_x = d['atm_param_cov_p']
        print('Covariance: J C_theta J^T ...')
        cov_y += np.einsum('nik,nkl,njl->nij', J, cov_x, J)

    # If pretrained model provided, could calculate reduced chi^2
    # with maximum-likelihood (mu, E) here.

    # \delta \mu (must be added in after transformation,
    #             due to possibly infinite terms).
    print('{:d} NaN parallaxes'.format(
        np.count_nonzero(np.isnan(d['parallax']))
    ))
    err_over_plx = d['parallax_err'] / d['parallax']
    print('Covariance: DM uncertainty term ...')
    cov_y[:,0,0] += (5./np.log(10.) * err_over_plx)**2.

    # Subtract distance modulus from m_G
    #dm = -5. * (np.log10(d['parallax']) - 2.)
    #dm_corr = 0.5 * err_over_plx**2 + 0.75 * err_over_plx**4
    #dm_corr_pct = np.percentile(dm_corr, [1., 5., 10., 50., 90., 95., 99.])
    #print(dm_corr_pct)

    print('Estimate DM ...')
    dm = 10. - 5.*np.log10(d['parallax'])# + 5./np.log(10.)*dm_corr
    y[:,0] -= dm

    # Don't attempt to predict M_G for poor plx/err or when plx < 0
    print('Filter out M_G for poor parallax measurements ...')
    idx = (
          (err_over_plx > 0.2)
        | (d['parallax'] < 1.e-8)
        | ~np.isfinite(d['parallax'])
        | ~np.isfinite(d['parallax_err'])
    )
    n_use = idx.size - np.count_nonzero(idx)
    print(r'Using {:d} of {:d} ({:.3f}%) of stellar parallaxes.'.format(
        n_use, idx.size, n_use/idx.size*100.
    ))
    cov_y[idx,0,0] = 99.**2
    y[idx,0] = np.nanmedian(y[:,0])

    # Reddenings
    print('Copy reddenings ...')
    r = np.empty((d.size,), dtype='f4')
    r[:] = d['r'][:]

    if pretrained_model is not None:
        # Update reddenings, based on vector R and (y_obs - y_pred).
        # Use provided reddenings as a prior.

        # First, need to calculate inv_cov_y
        print('Invert C_y matrices ...')
        inv_cov_y = np.stack([np.linalg.inv(c) for c in cov_y])

        # Calculate posterior on reddening
        print('Calculate posterior on reddening ...')
        sigma_r = d['r_err']
        y_pred = predict_y(pretrained_model, x_p)
        R = extract_R(pretrained_model)
        r_pred, r_var, chisq = update_reddenings(
            R, inv_cov_y, y, y_pred,
            r, sigma_r
        )
        print('chisq =', chisq)

        # Calculate d.o.f. of each star
        print('Calculate d.o.f. of each star ...')
        n_dof = np.zeros(d.size, dtype='i4')
        for k in range(n_bands):
            n_dof += (cov_y[:,k,k] < 99.**2).astype('i4')
        print('n_dof =', n_dof)

        # Calculate reduced chi^2 for each star
        print('Calculate chi^2/d.o.f. for each star ...')
        rchisq = chisq / (n_dof - 1.)
        pct = (0., 1., 10., 50., 90., 99., 100.)
        rchisq_pct = np.percentile(rchisq[np.isfinite(rchisq)], pct)
        print('chi^2/dof percentiles:')
        for p,rc in zip(pct,rchisq_pct):
            print(rf'  {p:.0f}% : {rc:.3g}')

        # Clip mean and variance of reddenings
        print('Clip reddenings and reddening variances ...')
        r_pred = np.clip(r_pred, 0., 10.) # TODO: Update upper limit?
        r_var = np.clip(r_var, 0.01**2, 10.**2)

        r[:] = r_pred

        # Reddening uncertainty term in covariance of y
        print('Covariance: reddening uncertainty term ...')
        cov_y += r_var[:,None,None] * np.outer(R, R)[None,:,:]
        
        # Filter on reduced chi^2
        if rchisq_max is not None:
            print('Filter on chi^2/d.o.f. ...')
            idx = np.isfinite(rchisq) & (rchisq > 0.) & (rchisq < rchisq_max)
            n_filt = np.count_nonzero(~idx)
            pct_filt = 100. * n_filt / idx.size
            print(
                rf'Filtering {n_filt:d} stars ({pct_filt:.3g}%) ' +
                rf'based on chi^2/dof > {rchisq_max:.1f}'
            )
            x = x[idx]
            x_p = x_p[idx]
            r = r[idx]
            y = y[idx]
            cov_y = cov_y[idx]

    # Cholesky transform of inverse covariance: L L^T = C^(-1).
    print('Cholesky transform of each stellar covariance matrix ...')
    LT = np.empty_like(cov_y)
    inv_cov_y = np.empty_like(cov_y)
    #LT = []
    #inv_cov_y = []
    for k,c in enumerate(cov_y):
        try:
            inv_cov_y[k] = np.linalg.inv(c)
            LT[k] = np.linalg.cholesky(inv_cov_y[k].T)
            #ic = np.linalg.inv(c)
            #LT.append(np.linalg.cholesky(ic).T)
            #inv_cov_y.append(ic)
        except np.linalg.LinAlgError as e:
            rho = get_corr_matrix(c)
            print('Offending correlation matrix:')
            print(np.array2string(
                rho[:6,:6],
                formatter={'float_kind':lambda z:'{: >7.4f}'.format(z)}
            ))
            print('Offending covariance matrix:')
            print(np.array2string(
                c[:6,:6],
                formatter={'float_kind':lambda z:'{: >9.6f}'.format(z)}
            ))
            print('Covariance matrix of (normed) atmospheric parameters:')
            print(d['atm_param_cov_p'][k])
            raise e

    #print('Stack L^T matrices ...')
    #LT = np.stack(LT)
    #print('Stack C^(-1) matrices ...')
    #inv_cov_y = np.stack(inv_cov_y)

    # L^T y
    print('Calculate L^T y ...')
    LTy = np.einsum('nij,nj->ni', LT, y)

    print('Gather inputs and outputs and return ...')
    inputs_outputs = {
        'x':x, 'x_p':x_p, 'r':r, 'y':y,
        'LT':LT, 'LTy':LTy,
        'cov_y':cov_y, 'inv_cov_y':inv_cov_y,
    }

    # Check that there are no NaNs or Infs in results
    for key in inputs_outputs:
        if np.any(~np.isfinite(inputs_outputs[key])):
            raise ValueError(f'NaNs or Infs detected in {key}.')

    return inputs_outputs


def predict_y(nn_model, x_p):
    inputs = nn_model.get_layer(name='atm_params').input
    outputs = nn_model.get_layer(name='mag_color').output
    mag_color_model = keras.Model(inputs, outputs)
    y = mag_color_model.predict(x_p)
    return y


def save_predictions(fname, nn_model, d_test, io_test):
    y_pred = predict_y(nn_model, io_test['x_p'])
    R = extract_R(nn_model)
    
    with h5py.File(fname, 'w') as f:
        f.create_dataset('/data', data=d_test, chunks=True,
                         compression='gzip', compression_opts=3)
        f.create_dataset('/y_obs', data=io_test['y'], chunks=True,
                         compression='gzip', compression_opts=3)
        f.create_dataset('/cov_y', data=io_test['cov_y'], chunks=True,
                         compression='gzip', compression_opts=3)
        f.create_dataset('/reddening', data=io_test['r'], chunks=True,
                         compression='gzip', compression_opts=3)
        f.create_dataset('/y_pred', data=y_pred, chunks=True,
                         compression='gzip', compression_opts=3)
        f.attrs['R'] = R


def extract_R(nn_model):
    R = nn_model.get_layer('extinction_reddening').get_weights()[0][0]
    #R = nn_model.get_layer('extinction').get_weights()[0][0]
    return R


def update_reddenings(R, inv_cov_y, y_obs, y_pred, r0, sigma_r):
    """
    Calculate posterior on reddening of each star, given
    difference between prediction and observation, and covariance
    matrix of the difference.
    
    Let n = # of bands, k = # of stars.

    Inputs:
        R (np.ndarray): Shape-(n,) array containing reddening vector.
        inv_cov_y (np.ndarray): Shape-(k,n,n) array containing
            covariance matrix of y_obs-y_pred for each star.
        y_obs (np.ndarray): Shape-(k,n) array containing observed
            magnitudes & colors for each star.
        y_pred (np.ndarray): Shape-(k,n) array containing predicted
            zero-reddening magnitudes & colors for each star.
        r0 (np.ndarray): Shape-(k,) array containing mean of prior on
            reddening for each star.
        sigma_r (np.ndarray): Shape-(k,) array containing std. dev. of
            prior on reddening for each star.

    Outputs:
        r_mean (np.ndarray): Shape-(k,) array containing mean posterior
            reddening of each star.
        r_var (np.ndarray): Shape-(k,) array containing variance of
            reddening posterior for each star.
        chisq (np.ndarray): Shape-(k,) array containing chi^2 of
            solution for each star.
    """
    print('Updating reddenings:')
    print('  * R^T C_y^(-1) ...')
    RT_Cinv = np.einsum('i,nij->nj', R.T, inv_cov_y)
    print('  * num = r_0/sigma_r^2 + [R^T C_y^(-1)] dy ...')
    num = r0/sigma_r**2 + np.einsum('ni,ni->n', RT_Cinv, y_obs - y_pred)
    print('  * den = [R^T C_y^(-1)] R + 1/sigma_r^2 ...')
    den = np.einsum('ni,i->n', RT_Cinv, R) + 1./sigma_r**2
    print('  * r_mean, r_var ...')
    r_mean = num / den
    r_var = 1. / den

    # Chi^2
    print('  * dy = y_pred + R <r> - y_obs ...')
    dy = y_pred + R[None,:]*r_mean[:,None] - y_obs
    print('  * chi^2: C_y^(-1) dy ...')
    chisq = np.einsum('nij,nj->ni', inv_cov_y, dy)
    print('  * chi^2: dy^T [C_y^(-1) dy] ...')
    chisq = np.einsum('ni,ni->n', dy, chisq)
    #chisq = np.einsum('ni,nij,nj->n', dy, inv_cov_y, dy)

    return r_mean, r_var, chisq


class ReddeningRegularizer(keras.regularizers.Regularizer):
    """
    Kernel regularizer that punishes negative entries in the
    reddening vector. Adapted from tf.keras.regularizers.L1L2.

    Arguments:
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        if not self.l1 and not self.l2:
            return K.constant(0.)
        regularization = 0.
        # Convert from reddening to extinction vector, assuming that
        # the weights x represent (G, X1-G, X2-G, X3-G, ...). This
        # operation turns x into (G+G, X1, X2, X3, ...). The first
        # entry is 2G instead of G, but the G component never goes
        # negative in practice.
        x = tf.math.add(x, x[...,0])
        # Only penalize negative values
        x = tf.math.minimum(x, 0.)
        if self.l1: # 1-norm regularization
            regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        if self.l2: # 2-norm regularization
            regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


def get_nn_model(n_hidden_layers=1, hidden_size=32, l2=1.e-3, n_bands=13):
    # f : \theta --> M
    atm = keras.Input(shape=(3,), name='atm_params')
    x = atm
    for i in range(n_hidden_layers):
        x = keras.layers.Dense(
            hidden_size,
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l2(l=l2)
        )(x)
    mag_color = keras.layers.Dense(n_bands, name='mag_color')(x)

    # Extinction/reddening
    red = keras.Input(shape=(1,), name='reddening')
    ext_red = keras.layers.Dense(
        n_bands,
        use_bias=False,
        kernel_regularizer=ReddeningRegularizer(l1=0.1),
        name='extinction_reddening'
    )(red)

    #B = np.identity(n_bands, dtype='f4')
    #B[1:,0] = -1.
    #B = tf.constant(B)
    #B = K.reshape(B, (-1, n_bands, n_bands))
    #ext_red = keras.layers.Dot((1,1), name='ext_red')([B, ext])

    y = keras.layers.Add(name='reddened_mag_color')([mag_color, ext_red])

    # Cholesky decomposition of inverse covariance matrix, L L^T = C^(-1)
    LT = keras.Input(shape=(n_bands,n_bands), name='LT')

    # Multiply y by L^T, since loss is given by |L^T (y_pred - y_obs)|^2
    LTy = keras.layers.Dot((2,1), name='LTy')([LT, y])

    # Compile model
    model = keras.Model(inputs=[atm,red,LT], outputs=LTy)
    model.compile(
        loss='mse',
        optimizer='Adam',
        metrics=['mse']
    )

    return model


def split_dataset(frac, *args):
    assert len(args) != 0

    n_tot = args[0].shape[0]
    idx = np.arange(n_tot)
    np.random.shuffle(idx)

    n = int(frac * n_tot)
    idx_left = idx[:n]
    idx_right = idx[n:]

    left, right = [], []

    for x in args:
        left.append(x[idx_left])
        right.append(x[idx_right])
    
    return left, right


def train_model(nn_model, io_train, epochs=100,
                checkpoint_fn='checkpoint', batch_size=32):
    checkpoint_fn = (
          'checkpoints/'
        + checkpoint_fn
        + '.e{epoch:03d}_vl{val_loss:.3f}.h5'
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_fn,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    inputs = [io_train['x_p'], io_train['r'], io_train['LT']]
    outputs = io_train['LTy']
    nn_model.fit(
        inputs, outputs,
        epochs=epochs,
        validation_split=0.25/0.9,
        callbacks=callbacks,
        batch_size=batch_size
    )


def diagnostic_plots(nn_model, io_test, d_test, suffix=None):
    if suffix is None:
        suff = ''
    else:
        suff = '_' + suffix
    
    inputs = [
        nn_model.get_layer(name='atm_params').input,
        nn_model.get_layer(name='reddening').input
    ]
    outputs = nn_model.get_layer(name='reddened_mag_color').output
    absmag_model = keras.Model(inputs, outputs)

    # Predict y for the test dataset
    test_pred = {
        'y': absmag_model.predict([
            io_test['x_p'],
            io_test['r']
        ]),
        'y_dered': absmag_model.predict([
            io_test['x_p'],
            np.zeros_like(io_test['r'])
        ])
    }
    test_pred['y_resid'] = io_test['y'] - test_pred['y']

    # Get the extinction vector
    R = extract_R(nn_model)
    R[1:] += R[0]
    print(
          'Extinction/reddening vector: ['
        + ' '.join(list(map('{:.3f}'.format,R)))
        + ']'
    )

    # Read out colors, magnitudes
    g = io_test['y'][:,3] + io_test['y'][:,0]
    ri = io_test['y'][:,4] - io_test['y'][:,5]
    gr = io_test['y'][:,3] - io_test['y'][:,4]
    g_pred = test_pred['y'][:,3] + test_pred['y'][:,0]
    ri_pred = test_pred['y'][:,4] - test_pred['y'][:,5]
    gr_pred = test_pred['y'][:,3] - test_pred['y'][:,4]
    g_pred_dered = test_pred['y_dered'][:,3] + test_pred['y_dered'][:,0]
    ri_pred_dered = test_pred['y_dered'][:,4] - test_pred['y_dered'][:,5]
    gr_pred_dered = test_pred['y_dered'][:,3] - test_pred['y_dered'][:,4]
    A_g = 0.25 * R[3]
    E_ri = 0.25 * (R[4] - R[5])
    E_gr = 0.25 * (R[3] - R[4])

    gaia_g = io_test['y'][:,0]
    bp_rp = io_test['y'][:,1] - io_test['y'][:,2]
    gaia_g_pred = test_pred['y'][:,0]
    bp_rp_pred = test_pred['y'][:,1] - test_pred['y'][:,2]
    gaia_g_pred_dered = test_pred['y_dered'][:,0]
    gaia_bp_rp_pred_dered = test_pred['y_dered'][:,1] - test_pred['y_dered'][:,2]
    A_gaia_g = 0.25 * R[0]
    E_bp_rp = 0.25 * (R[1] - R[2])

    print('g =', g)
    print('ri =', ri)
    print('gr =', gr)
    print('gaia_g =', gaia_g)
    print('bp_rp =', bp_rp)

    # Plot HRD
    params = {
        'density': (None, r'$N$', (None, None)),
        'teff': (d_test['atm_param'][:,0], r'$T_{\mathrm{eff}}$', (4000., 8000.)),
        'logg': (d_test['atm_param'][:,1], r'$\log \left( g \right)$', (0., 5.)),
        'mh': (d_test['atm_param'][:,2], r'$\left[ \mathrm{M} / \mathrm{H} \right]$', (-2.5, 0.5))
    }

    plot_spec = [
        {
            'colors': [(1,2), (4,5)],
            'mag': 0
        },
        {
            'colors': [(3,4), (4,5)],
            'mag': 0
        }
    ]

    idx_goodobs = np.isfinite(d_test['mag_err'])
    idx_goodobs &= (np.abs(io_test['cov_y'][:,0,0]) < 90.)[:,None]
    idx_goodobs = idx_goodobs.T

    def scatter_or_hexbin(ax, x, y, c, vmin, vmax, extent):
        if p == 'density':
            im = ax.hexbin(
                x, y,
                extent=extent,
                bins='log',
                rasterized=True
            )
        else:
            im = ax.scatter(
                x,
                y,
                c=c,
                edgecolors='none',
                alpha=0.1,
                vmin=vmin,
                vmax=vmax,
                rasterized=True
            )
        return im

    def get_lim(*args, **kwargs):
        expand = kwargs.get('expand', 0.4)
        expand_low = kwargs.get('expand_low', expand)
        expand_high = kwargs.get('expand_high', expand)
        pct = kwargs.get('pct', 1.)
        lim = [np.inf, -np.inf]
        for a in args:
            a0,a1 = np.nanpercentile(a, [pct, 100.-pct])
            lim[0] = min(a0, lim[0])
            lim[1] = max(a1, lim[1])
        w = lim[1] - lim[0]
        lim[0] -= expand_low * w
        lim[1] += expand_high * w
        return lim

    labels = ['G', 'BP', 'RP', 'g', 'r', 'i', 'z', 'y']

    for ps in plot_spec:
        mag_label = r'$M_{{ {} }}$'.format(labels[ps['mag']])
        mag_obs = io_test['y'][:,ps['mag']]
        mag_pred = test_pred['y'][:,ps['mag']]
        mag_pred_dered = test_pred['y_dered'][:,ps['mag']]
        A_vec = 0.25 * R[ps['mag']]
        print('mag_pred:',mag_pred)

        if ps['mag'] != 0:
            mag_obs += io_test['y'][:,0]
            mag_pred += io_test['y'][:,0]
            mag_pred_dered += io_test['y'][:,0]
            A_vec += 0.25 * R[0]

        color_labels = []
        colors_obs = []
        colors_pred = []
        colors_pred_dered = []
        idx_colors_obs = []
        E_vec = []
        for i1,i2 in ps['colors']:
            color_labels.append(r'${} - {}$'.format(labels[i1], labels[i2]))
            colors_obs.append(io_test['y'][:,i1] - io_test['y'][:,i2])
            colors_pred.append(test_pred['y'][:,i1] - test_pred['y'][:,i2])
            colors_pred_dered.append(
                test_pred['y_dered'][:,i1] - test_pred['y_dered'][:,i2]
            )
            idx_colors_obs.append(idx_goodobs[i1] & idx_goodobs[i2])
            E_vec.append(0.25 * (R[i1] - R[i2]))

        mag_lim = get_lim(
            mag_obs[idx_goodobs[ps['mag']]],
            pct=2.
        )[::-1]
        color_lim = [
            get_lim(c[idx_colors_obs[k]], expand_low=0.5, expand_high=0.4)
            for k,c in enumerate(colors_obs)
        ]
        
        for p in params.keys():
            c, label, (vmin,vmax) = params[p]
            
            fig = plt.figure(figsize=(14,4.5), dpi=150)
            fig.patch.set_alpha(0.)
            gs = GridSpec(
                1,4,
                width_ratios=[1,1,1,0.1],
                left=0.07, right=0.93,
                bottom=0.10, top=0.92
            )
            ax_obs = fig.add_subplot(gs[0,0], facecolor='gray')
            ax_pred = fig.add_subplot(gs[0,1], facecolor='gray')
            ax_dered = fig.add_subplot(gs[0,2], facecolor='gray')
            cax = fig.add_subplot(gs[0,3], facecolor='w')

            extent = color_lim[0] + mag_lim

            idx = (
                  idx_goodobs[ps['mag']]
                & idx_goodobs[ps['colors'][0][0]]
                & idx_goodobs[ps['colors'][0][1]]
            )
            im = scatter_or_hexbin(
                ax_obs,
                colors_obs[0][idx],
                mag_obs[idx],
                c if c is None else c[idx],
                vmin, vmax,
                extent
                #(-0.3,1.0,11.5,-2.0)
            )

            ax_obs.set_xlim(color_lim[0])
            ax_obs.set_ylim(mag_lim)
            ax_obs.set_xlabel(color_labels[0])
            ax_obs.set_ylabel(mag_label)
            ax_obs.grid('on', alpha=0.3)
            ax_obs.set_title(r'$\mathrm{Observed}$')

            im = scatter_or_hexbin(
                ax_pred,
                colors_pred[0],
                mag_pred,
                c,
                vmin, vmax,
                extent
            )

            ax_pred.set_xlim(color_lim[0])
            ax_pred.set_ylim(mag_lim)
            ax_pred.set_xlabel(color_labels[0])
            ax_pred.grid('on', alpha=0.3)
            ax_pred.set_title(r'$\mathrm{Predicted}$')

            im = scatter_or_hexbin(
                ax_dered,
                colors_pred_dered[0],
                mag_pred_dered,
                c,
                vmin, vmax,
                extent
            )

            ax_dered.set_xlim(color_lim[0])
            ax_dered.set_ylim(mag_lim)
            ax_dered.set_xlabel(color_labels[0])
            ax_dered.grid('on', alpha=0.3)
            ax_dered.set_title(r'$\mathrm{Predicted+Dereddened}$')

            ax_dered.annotate(
                '',
                xy=(0.35+E_vec[0], 1.+A_vec),
                xytext=(0.35, 1.),
                arrowprops=dict(color='r', width=1., headwidth=5., headlength=5.)
            )

            cb = fig.colorbar(im, cax=cax)
            cb.set_label(label)
            cb.set_alpha(1.)
            cb.draw_all()

            cm_desc = '{}_vs_{}{}'.format(
                labels[ps['mag']],
                labels[ps['colors'][0][0]],
                labels[ps['colors'][0][1]]
            )
            fig.savefig(
                'plots/nn_predictions_'+cm_desc+'_'+p+suff+'.svg',
                dpi=150,
                facecolor=fig.get_facecolor(),
                edgecolor='none'
            )
            plt.close(fig)

            # Color-color diagrams
            fig = plt.figure(figsize=(14,4.5), dpi=150)
            fig.patch.set_alpha(0.)
            gs = GridSpec(
                1,4,
                width_ratios=[1,1,1,0.1],
                left=0.07, right=0.93,
                bottom=0.10, top=0.92
            )
            ax_obs = fig.add_subplot(gs[0,0], facecolor='gray')
            ax_pred = fig.add_subplot(gs[0,1], facecolor='gray')
            ax_dered = fig.add_subplot(gs[0,2], facecolor='gray')
            cax = fig.add_subplot(gs[0,3], facecolor='w')

            extent = color_lim[0] + color_lim[1]

            idx = (
                  idx_goodobs[ps['colors'][0][0]]
                & idx_goodobs[ps['colors'][0][1]]
                & idx_goodobs[ps['colors'][1][0]]
                & idx_goodobs[ps['colors'][1][1]]
            )

            im = scatter_or_hexbin(
                ax_obs,
                colors_obs[0][idx],
                colors_obs[1][idx],
                c if c is None else c[idx],
                vmin, vmax,
                extent
                #(-0.2,1.5,-0.15,0.8)
            )
            ax_obs.set_xlim(color_lim[0])
            ax_obs.set_ylim(color_lim[1])
            ax_obs.set_xlabel(color_labels[0], fontsize=14)
            ax_obs.set_ylabel(color_labels[1], fontsize=14)
            ax_obs.grid('on', alpha=0.3)
            ax_obs.set_title(r'$\mathrm{Observed}$')

            im = scatter_or_hexbin(
                ax_pred,
                colors_pred[0][idx],
                colors_pred[1][idx],
                c if c is None else c[idx],
                #c,
                vmin, vmax,
                extent
            )
            ax_pred.set_xlim(color_lim[0])
            ax_pred.set_ylim(color_lim[1])
            ax_pred.set_xlabel(color_labels[0], fontsize=14)
            ax_pred.grid('on', alpha=0.3)
            ax_pred.set_title(r'$\mathrm{Predicted}$')

            im = scatter_or_hexbin(
                ax_dered,
                colors_pred_dered[0][idx],
                colors_pred_dered[1][idx],
                c if c is None else c[idx],
                #c,
                vmin, vmax,
                extent
            )
            ax_dered.set_xlim(color_lim[0])
            ax_dered.set_ylim(color_lim[1])
            ax_dered.set_xlabel(color_labels[0], fontsize=14)
            ax_dered.grid('on', alpha=0.3)
            ax_dered.set_title(r'$\mathrm{Predicted+Dereddened}$')

            ax_dered.annotate(
                '',
                xy=(0.4+E_vec[0], 0.3+E_vec[1]),
                xytext=(0.4, 0.3),
                arrowprops=dict(color='r', width=1., headwidth=5., headlength=5.)
            )

            cb = fig.colorbar(im, cax=cax)
            cb.set_label(label, fontsize=14)
            cb.set_alpha(1.)
            cb.draw_all()

            cc_desc = '{}{}_vs_{}{}'.format(
                labels[ps['colors'][0][0]],
                labels[ps['colors'][0][1]],
                labels[ps['colors'][1][0]],
                labels[ps['colors'][1][1]]
            )
            fig.savefig(
                'plots/test_'+cc_desc+'_'+p+suff+'.svg',
                dpi=150,
                facecolor=fig.get_facecolor(),
                edgecolor='none'
            )
            plt.close(fig)

    # Plot histogram of reddening residuals
    dr = io_test['r'] - d_test['r']
    fig = plt.figure(figsize=(8,5), dpi=150)
    ax = fig.add_subplot(1,1,1)
    ax.hist(dr, range=(-0.15, 0.15), bins=50)
    dr_mean = np.nanmean(dr)
    dr_std = np.std(dr)
    dr_skew = scipy.stats.moment(dr, moment=3, nan_policy='omit')
    dr_txt = r'$\Delta E = {:+.3f} \pm {:.3f}$'.format(dr_mean, dr_std)
    dr_skew /= (dr_std**1.5 + 1.e-5)
    dr_txt += '\n' + r'$\tilde{{\mu}}_3 = {:+.3f}$'.format(dr_skew)
    ax.text(0.05, 0.95, dr_txt, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel(
        r'$\Delta E \ \left( \mathrm{estimated} - \mathrm{Bayestar19} \right)$',
        fontsize=14
    )
    fig.savefig('plots/test_dE'+suff+'.svg', dpi=150)
    plt.close(fig)


def calc_dmag_color_dtheta(nn_model, x_p):
    m = keras.Model(
        inputs=nn_model.get_layer(name='atm_params').input,
        outputs=nn_model.get_layer(name='mag_color').output
    )
    with tf.GradientTape() as g:
        x_p = tf.constant(x_p)
        g.watch(x_p)
        mag_color = m(x_p)
    J = g.batch_jacobian(mag_color, x_p).numpy()
    return J


def main():
    # Load stellar data
    print('Loading data ...')
    fname = 'data/dr16_ddpayne_data.h5'
    d = load_data(fname)

    # (training+validation) / test split
    # Fix random seed (same split every run)
    np.random.seed(7)
    (d_train,), (d_test,) = split_dataset(0.9, d)
    np.random.shuffle(d_train) # Want d_train to be in random order
    print(f'{d_train.size: >10d} training/validation stars.')
    print(f'{d_test.size: >10d} test stars.')

    # Load/create neural network
    nn_name = 'dr16_ddpayne2'
    n_hidden = 2
    nn_model = get_nn_model(n_hidden_layers=n_hidden, l2=1.e-4)
    #nn_model = keras.models.load_model(
    #    'models/{:s}_{:d}hidden_it0.h5'.format(nn_name, n_hidden),
    #    custom_objects={'ReddeningRegularizer':ReddeningRegularizer}
    #)
    nn_model.summary()

    # Iteratively update dM/dtheta contribution to uncertainties,
    # reddening estimates and reduced chi^2 cut, and retrain.
    n_iterations = 15
    
    # On GPU, use large batch sizes for memory transfer efficiency
    batch_size = 1024

    rchisq_max_init = 100.
    rchisq_max_final = 5.
    rchisq_max = np.exp(np.linspace(
        np.log(rchisq_max_init),
        np.log(rchisq_max_final),
        n_iterations-1
    ))
    rchisq_max = [None] + rchisq_max.tolist()
    print('chi^2/dof = {}'.format(rchisq_max))

    for k in range(n_iterations):
        # Transform data to inputs and outputs
        # On subsequent iterations, inflate errors using
        # gradients dM/dtheta from trained model, and derive new
        # estimates of the reddenings of the stars.
        t0 = time()
        io_train = get_inputs_outputs(
            d_train,
            pretrained_model=None if k == 0 else nn_model,
            rchisq_max=rchisq_max[k]
        )
        io_test = get_inputs_outputs(
            d_test,
            pretrained_model=None if k == 0 else nn_model
        )
        t1 = time()
        print(f'Time elapsed to prepare data: {t1-t0:.2f} s')

        # Set learning rate based on the iteration
        lr = 0.001 * np.exp(-0.2*k)
        print('learning rate = {}'.format(K.get_value(nn_model.optimizer.lr)))
        print('setting learning rate to {}'.format(lr))
        K.set_value(nn_model.optimizer.lr, lr)
        
        # Train the model
        print('Iteration {} of {}.'.format(k+1, n_iterations))
        t0 = time()
        train_model(
            nn_model,
            io_train,
            epochs=25,
            checkpoint_fn='{:s}_{:d}hidden_it{:d}'.format(
                nn_name, n_hidden, k
            ),
            batch_size=batch_size
        )
        t1 = time()
        print(f'Time elapsed to train: {t1-t0:.2f} s')
        nn_model.save(
            'models/{:s}_{:d}hidden_it{:d}.h5'.format(
                nn_name, n_hidden, k
            )
        )
        #nn_model = keras.models.load_model(
        #    'models/{:s}_{:d}hidden_it{:d}.h5'.format(nn_name, n_hidden, k),
        #    custom_objects={'ReddeningRegularizer':ReddeningRegularizer}
        #)

        # Plot initial results
        print('Diagnostic plots ...')
        t0 = time()
        diagnostic_plots(
            nn_model,
            io_test,
            d_test,
            #io_train,
            #d_train,
            suffix='{:s}_{:d}hidden_it{:d}'.format(nn_name, n_hidden, k)
        )
        t1 = time()
        print(f'Time elapsed to make plots: {t1-t0:.2f} s')

    fname = 'data/predictions_{:s}_{:d}hidden_it{:d}.h5'.format(
        nn_name, n_hidden, n_iterations-1
    )
    save_predictions(fname, nn_model, d_test, io_test)

    return 0

if __name__ == '__main__':
    main()

