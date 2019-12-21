#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import scipy.stats
import h5py
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from glob import glob


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

    # Add error floor into magnitude uncertainties
    err_floor = 0.02
    d['ps1_mag_err'] = np.sqrt(d['ps1_mag_err']**2 + err_floor**2)
    for b in ('g', 'bp', 'rp'):
        d['gaia_'+b+'_mag_err'] = np.sqrt(
              d['gaia_'+b+'_mag_err']**2
            + err_floor**2
        )

    # Add error floor into reddening uncertainties
    r_err_floor = 0.05
    r_err_scale = 0.10
    b19_err = np.sqrt(
          b19_err**2
        + r_err_floor**2
        + (r_err_scale*b19_err)**2
    )
    
    # Filter out stars with no reddening estimates
    idx = np.isfinite(b19)
    d = d[idx]
    b19 = b19[idx]
    b19_err = b19_err[idx]

    return d, b19, b19_err


def collapse_atm_params(d, b19, b19_err):
    dtype = [
        ('b19', 'f4'),
        ('b19_err', 'f4'),
        ('SFD', 'f4'),
        ('ps1_mag', '5f4'),
        ('ps1_mag_err', '5f4'),
        ('gaia_g_mag', 'f4'),
        ('gaia_g_mag_err', 'f4'),
        ('gaia_bp_mag', 'f4'),
        ('gaia_bp_mag_err', 'f4'),
        ('gaia_rp_mag', 'f4'),
        ('gaia_rp_mag_err', 'f4'),
        ('parallax', 'f4'),
        ('parallax_err', 'f4'),
        ('teff', 'f4'),
        ('logg', 'f4'),
        ('feh', 'f4'),
        ('teff_err', 'f4'),
        ('logg_err', 'f4'),
        ('feh_err', 'f4')
    ]
    
    d_simp = np.empty(d.size, dtype=dtype)
    d_simp['b19'] = b19
    d_simp['b19_err'] = b19_err
    for key,_ in dtype[2:-6]:
        d_simp[key] = d[key]

    for source in ('galah', 'apogee', 'ddpayne'):
        idx = np.isfinite(d[source+'_teff'])
        for key,_ in dtype[-6:]:
            d_simp[key][idx] = d[source+'_'+key][idx]

    idx = (
          (d_simp['teff_err'] < 200.)
        & (d_simp['logg_err'] < 0.5)
        & (d_simp['feh_err'] < 0.5)
    )
    d_simp = d_simp[idx]

    return d_simp


def get_corr_matrix(cov):
    rho = cov.copy()
    sqrt_cov_diag = np.sqrt(cov[np.diag_indices(cov.shape[0])])
    rho /= sqrt_cov_diag[:,None]
    rho /= sqrt_cov_diag[None,:]
    rho[np.diag_indices(cov.shape[0])] = sqrt_cov_diag
    return rho


def get_inputs_outputs(d, normalizers=None, pretrained_model=None):
    n_bands = 8 # Gaia (G, BP, RP), PS1 (grizy)

    # Stellar spectroscopic parameters
    x = np.empty((d.size,3), dtype='f4')
    x[:,0] = d['teff']
    x[:,1] = d['logg']
    x[:,2] = d['feh']

    # Normalize x (zero median, unit variance)
    if normalizers is None:
        norm = {'x': DataNormalizer(x)}
    else:
        norm = normalizers
    x_p = norm['x'].normalize(x)

    # Magnitudes
    y = np.empty((d.size,n_bands), dtype='f4')
    y[:,0] = d['gaia_g_mag']
    y[:,1] = d['gaia_bp_mag']
    y[:,2] = d['gaia_rp_mag']
    y[:,3:] = d['ps1_mag']

    #idx_y_bad = np.isnan(y) # Replace NaN magnitudes with zero
    #y[idx_y_bad] = 0.

    for b in range(n_bands):
        idx = np.isnan(y[:,b])
        y[idx,b] = np.median(y[~idx,b])

    # Covariance of y
    cov_y = np.zeros((d.size,n_bands,n_bands), dtype='f4')

    # \delta m
    cov_y[:,0,0] += d['gaia_g_mag_err']**2
    cov_y[:,1,1] += d['gaia_bp_mag_err']**2
    cov_y[:,2,2] += d['gaia_rp_mag_err']**2
    for k in range(5):
        cov_y[:,3+k,3+k] += d['ps1_mag_err'][:,k]**2

    idx = np.isnan(cov_y) # Replace NaN errors with large number
    cov_y[idx] = 100.**2.

    # \delta A
    # \partial M / \partial \theta

    # Transform both y and its covariance
    B = np.identity(n_bands, dtype='f4')
    B[1:,0] = -1.

    y = np.einsum('ij,nj->ni', B, y) # y' = B y
    cov_y = np.einsum('ik,nkl,jl->nij', B, cov_y, B) # C' = B C B^T

    # Add in dM/dtheta term
    # TODO: Off-diagonal elements of theta covariance matrix
    if pretrained_model is not None:
        J = calc_dmag_color_dtheta(pretrained_model, x_p)
        cov_x = np.zeros((x.shape[0],3,3), dtype='f4')
        cov_x[:,0,0] = (d['teff_err'] / norm['x']._sigma[0])**2
        cov_x[:,1,1] = (d['logg_err'] / norm['x']._sigma[1])**2
        cov_x[:,2,2] = (d['feh_err'] / norm['x']._sigma[2])**2
        dcov_y = np.einsum('nik,nkl,njl->nij', J, cov_x, J)
        cov_y += dcov_y

    # \delta \mu (must be added in after transformation,
    #             due to possibly infinite terms).
    err_over_plx = d['parallax_err'] / d['parallax']
    cov_y[:,0,0] += (5./np.log(10.) * err_over_plx)**2.

    # Subtract distance modulus from m_G
    dm = -5. * (np.log10(d['parallax']) - 2.)
    y[:,0] -= dm

    # Don't attempt to predict M_G when plx/err < 4 or plx < 0
    idx = (err_over_plx > 0.25) | (d['parallax'] < 1.e-5)
    cov_y[idx,0,0] = 100.**2#np.inf
    y[idx,0] = 0.

    # Reddenings
    r = np.empty((d.size,), dtype='f4')
    r[:] = d['b19']

    if pretrained_model is not None:
        # Update reddenings, based on vector R and (y_obs - y_pred).
        # Use provided reddenings as a prior.

        # First, need to calculate inv_cov_y
        inv_cov_y = np.stack([np.linalg.inv(c) for c in cov_y])

        sigma_r = d['b19_err']
        y_pred = predict_y(pretrained_model, x_p)
        R = extract_R(pretrained_model)
        r_pred, r_var = update_reddenings(
            R, inv_cov_y, y, y_pred,
            r, sigma_r
        )

        # Clip mean and variance of reddenings
        r_pred = np.clip(r_pred, 0., 10.) # TODO: Update upper limit?
        r_var = np.clip(r_var, 0.05**2, 10.**2)

        print('Delta r:')
        print(r_pred - r)
        print('r_pred:')
        print(r_pred)
        print('r_std:')
        print(np.sqrt(r_var))

        r[:] = r_pred
        
        # Reddening uncertainty term in covariance of y
        cov_y += r_var[:,None,None] * np.outer(R, R)[None,:,:]

    # Cholesky transform of inverse covariance: L L^T = C^(-1).
    LT = []
    inv_cov_y = []
    for k,c in enumerate(cov_y):
        #try:
        ic = np.linalg.inv(c)
        LT.append(np.linalg.cholesky(ic).T)
        inv_cov_y.append(ic)
        #except np.linalg.LinAlgError as e:
        #    print(d['teff_err'][k], d['logg_err'][k], d['feh_err'][k])
        #    rho = get_corr_matrix(c)
        #    print(np.array2string(
        #        rho[:6,:6],
        #        formatter={'float_kind':lambda z:'{: >7.4f}'.format(z)}
        #    ))
        #    raise e

    LT = np.stack(LT)
    inv_cov_y = np.stack(inv_cov_y)

    # L^T y
    LTy = np.einsum('nij,nj->ni', LT, y)

    inputs_outputs = {
        'x':x, 'x_p':x_p, 'r':r, 'y':y,
        'LT':LT, 'LTy':LTy,
        'cov_y':cov_y, 'inv_cov_y':inv_cov_y,
    }

    return inputs_outputs, norm


def predict_y(nn_model, x_p):
    inputs = nn_model.get_layer(name='atm_params').input
    outputs = nn_model.get_layer(name='mag_color').output
    mag_color_model = keras.Model(inputs, outputs)
    y = mag_color_model.predict(x_p)
    return y


def extract_R(nn_model):
    R = nn_model.get_layer('extinction_reddening').get_weights()[0][0]
    return R


def update_reddenings(R, inv_cov_y, y_obs, y_pred, r0, sigma_r):
    RT_Cinv = np.einsum('i,nij->nj', R.T, inv_cov_y)
    num = r0/sigma_r**2 + np.einsum('ni,ni->n', RT_Cinv, y_obs - y_pred)
    den = np.einsum('ni,i->n', RT_Cinv, R) + 1./sigma_r**2
    r_mean = num / den
    r_var = 1. / den
    return r_mean, r_var


class DataNormalizer(object):
    def __init__(self, x, scale_only=False):
        if scale_only:
            self._mu = 0.
        else:
            self._mu = np.median(x, axis=0)
        self._sigma = np.std(x, axis=0)

    def normalize(self, x, scale_only=False):
        if scale_only:
            return x/self._sigma
        return (x-self._mu)/self._sigma

    def unnormalize(self, xp, scale_only=False):
        if scale_only:
            return xp*self._sigma
        return xp*self._sigma + self._mu

    def get_isig_norm(self):
        isig_norm = DataNormalizer([])
        isig_norm._mu = 0.
        isig_norm._sigma = 1. / self._sigma
        return isig_norm


def get_nn_model(n_hidden_layers=1, hidden_size=32, l2=1.e-3, n_bands=8):
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
        #kernel_constraint=keras.constraints.NonNeg(), # TODO: Relax (esp. for colors)?
        name='extinction_reddening'
    )(red)

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


def train_model(nn_model, io_train, epochs=100, checkpoint_fn='checkpoint'):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/'+checkpoint_fn+'.e{epoch:03d}_vl{val_loss:.3f}.h5',
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
        validation_split=0.25,
        callbacks=callbacks
    )


def diagnostic_plots(nn_model, io_test, d_test, normalizers, suffix=None):
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
    R = nn_model.get_layer('extinction_reddening').get_weights()[0][0]
    R[1:] += R[0]
    print('Extinction/reddening vector:', R)

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

    # Plot HRD
    params = {
        'density': (None, r'$N$', (None, None)),
        'teff': (d_test['teff'], r'$T_{\mathrm{eff}}$', (4000., 8000.)),
        'logg': (d_test['logg'], r'$\log \left( g \right)$', (0., 5.)),
        'feh': (d_test['feh'], r'$\left[ \mathrm{Fe} / \mathrm{H} \right]$', (-2.5, 0.5))
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

    idx_goodobs = [
        ~np.isnan(d_test['gaia_g_mag_err']),
        ~np.isnan(d_test['gaia_bp_mag_err']),
        ~np.isnan(d_test['gaia_rp_mag_err'])
    ]
    idx_goodobs += [~np.isnan(d_test['ps1_mag_err'][:,k]) for k in range(5)]
    for idx in idx_goodobs:
        idx &= (np.abs(io_test['y'][:,0]) > 1.e-5)

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
        pct = kwargs.get('pct', 1.)
        lim = [np.inf, -np.inf]
        for a in args:
            a0,a1 = np.nanpercentile(a, [pct, 100.-pct])
            lim[0] = min(a0, lim[0])
            lim[1] = max(a1, lim[1])
        w = lim[1] - lim[0]
        lim[0] -= expand*w
        lim[1] += expand*w
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

        mag_lim = get_lim(mag_obs[idx_goodobs[ps['mag']]], pct=2.)[::-1]
        color_lim = [
            get_lim(c[idx_colors_obs[k]]) for k,c in enumerate(colors_obs)
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

            print(len(colors_obs))
            print(colors_obs[0].shape)
            idx = (
                  idx_goodobs[ps['mag']]
                & idx_goodobs[ps['colors'][0][0]]
                & idx_goodobs[ps['colors'][0][1]]
            )
            print(idx)
            print(np.count_nonzero(idx))
            print(colors_obs[0][idx])
            print(mag_obs[idx])
            print(extent)
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

            im = scatter_or_hexbin(
                ax_pred,
                colors_pred[0],
                colors_pred[1],
                c,
                vmin, vmax,
                extent
            )
            ax_pred.set_xlim(color_lim[0])
            ax_pred.set_ylim(color_lim[1])
            ax_pred.set_xlabel(color_labels[0], fontsize=14)
            ax_pred.grid('on', alpha=0.3)

            im = scatter_or_hexbin(
                ax_dered,
                colors_pred_dered[0],
                colors_pred_dered[1],
                c,
                vmin, vmax,
                extent
            )
            ax_dered.set_xlim(color_lim[0])
            ax_dered.set_ylim(color_lim[1])
            ax_dered.set_xlabel(color_labels[0], fontsize=14)
            ax_dered.grid('on', alpha=0.3)

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
    dr = io_test['r'] - d_test['b19']
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
    fnames = glob('data/combined_data_*to*.h5')
    d,b19,b19_err = load_data(fnames)
    d = collapse_atm_params(d, b19, b19_err)

    # (training+validation) / test split
    # Fix random seed (same split every run)
    np.random.seed(7)
    (d_train,), (d_test,) = split_dataset(0.9, d)

    # Load/create neural network
    nn_name = 'combined_rerr'
    n_hidden = 2
    nn_model = get_nn_model(n_hidden_layers=n_hidden, l2=1.e-4)
    #nn_model = keras.models.load_model(
    #    'models/{:s}_{:d}hidden_it0.h5'.format(nn_name, n_hidden)
    #)
    nn_model.summary()

    # Iteratively update dM/dtheta contribution to uncertainties, and retrain
    n_iterations = 10

    for k in range(n_iterations):
        # Transform data to inputs and outputs
        # On subsequent iterations, inflate errors using
        # gradients dM/dtheta from trained model, and derive new
        # estimates of the reddenings of the stars.
        io_train,normalizers = get_inputs_outputs(
            d_train,
            pretrained_model=None if k == 0 else nn_model
        )
        io_test,_ = get_inputs_outputs(
            d_test,
            normalizers=normalizers,
            pretrained_model=None if k == 0 else nn_model
        )

        ## Set learning rate based on the iteration
        lr = 0.001 * np.exp(-0.2*k)
        print('learning rate = {}'.format(K.get_value(nn_model.optimizer.lr)))
        print('setting learning rate to {}'.format(lr))
        K.set_value(nn_model.optimizer.lr, lr)
        
        # Train the model
        train_model(
            nn_model,
            io_train,
            epochs=50,
            checkpoint_fn='{:s}_{:d}hidden_it{:d}'.format(
                nn_name, n_hidden, k
            )
        )
        nn_model.save(
            'models/{:s}_{:d}hidden_it{:d}.h5'.format(
                nn_name, n_hidden, k
            )
        )
        #nn_model = keras.models.load_model(
        #    'models/{:s}_{:d}hidden_it{:d}.h5'.format(nn_name, n_hidden, k)
        #)

        # Plot initial results
        diagnostic_plots(
            nn_model,
            io_test,
            d_test,
            normalizers,
            suffix='{:s}_{:d}hidden_it{:d}'.format(nn_name, n_hidden, k)
        )

    return 0

if __name__ == '__main__':
    main()

