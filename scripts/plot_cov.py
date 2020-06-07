#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

from corner import corner

from train_model_all import load_data, get_inputs_outputs, get_corr_matrix
#from train_model_dr16 import (load_data, get_inputs_outputs,
#                              get_corr_matrix, ReddeningRegularizer)
from tensorflow import keras

from plot_covariance_ellipses import plot_covariance


def plot_cov(cov, n_draws=1000, axes=None, mu=None, labels=None):
    n_dim = cov.shape[0]
    assert cov.shape == (n_dim, n_dim)
    
    # Draw samples from distribution
    if mu is None:
        y0 = np.zeros(n_dim)
    else:
        y0 = mu
    y = np.random.multivariate_normal(y0, cov, size=n_draws)
    
    # Select axes to plot
    if labels is None:
        y_labels = [f'$y_{{ {i} }}$' for i in range(n_dim)]
    else:
        y_labels = labels
    
    if axes is not None:
        y = y[:,axes]
        y_labels = [y_labels[i] for i in axes]
    
    kw = {}
    if np.allclose(y-y0[None,:], np.zeros_like(y)):
        kw['range'] = [(yy-1., yy+1.) for yy in y0]
        print('    ~ y = y_0')
    
    # Corner plot of samples
    fig = corner(
        y,
        labels=y_labels,
        show_titles=True,
        alpha=0.,
        title_fmt='.3f',
        **kw
    )
    
    return fig


def latexify_matrix(m, fmt='{: >+8.2f}'):
    txt = r'\begin{pmatrix} '
    for i,row in enumerate(m):
        if i != 0:
            txt += r' \\ '
        
        for j,x in enumerate(row):
            if j != 0:
                txt += ' & '
            
            txt += fmt.format(x)
    txt += r' \end{pmatrix}'
    return txt


def main():
    #R = np.ones(5)
    #cov = 0.1*np.identity(5) + R[None,:]*R[:,None]
    #print(cov)
    
    print('Loading data ...')
    fname = 'data/apogee_lamost_galah_data.h5'
    d = load_data(fname)
    
    np.random.shuffle(d) # Want d to be in random order
    d = d[:1000] # Select random subset of stars
    
    print('Loading model ...')
    nn_name = 'theta_dep_red_recalc2'
    n_hidden = 2
    nn_model = keras.models.load_model(
        'models/{:s}_{:d}hidden_it0.h5'.format(nn_name, n_hidden),
        #custom_objects={'ReddeningRegularizer':ReddeningRegularizer}
    )
    
    print('Calculating covariance matrices ...')
    io_dict = get_inputs_outputs(
        d,
        pretrained_model=nn_model,
        return_cov_components=True
    )
    
    print('Plotting covariance matrices ...')
    bands = ['BP', 'RP'] + list('grizyJH') + ['K_s', 'W_1', 'W_2']
    labels = ['$G$'] + [f'${s}-G$' for s in bands]
    
    sigma_r = np.sqrt(io_dict['r_var'])
    r0 = io_dict['r']

    idx = np.argsort(io_dict['rchisq'])
    idx = list(range(10)) + idx[:10].tolist() + idx[-10:].tolist()
    
    for n,i in enumerate(idx):
        print(f' - {n} (of {len(idx)})')
        
        y0 = io_dict['y'][i]
        
        cov_labels = [
            r'$\mathrm{total}$',
            r'$\left( \delta_{\theta} \cdot \nabla_{\theta} \right) \vec{M}$',
            r'$\hat{E} \left( \delta_{\theta} \cdot \nabla_{\theta} \right) \vec{R}$',
            r'$\delta E \vec{R}$',
            r'$\delta \mu$',
            r'$\delta \vec{m}$'
        ]
        cov_list = [
            io_dict['cov_y'][i],
            io_dict['cov_comp']['dM/dtheta'][i],
            io_dict['cov_comp']['dA/dtheta'][i],
            io_dict['cov_comp']['r'][i],
            io_dict['cov_comp']['dm'][i],
            io_dict['cov_comp']['delta_m'][i]
        ]
        dim_labels = labels
        linestyles = [':'] + ['-'] * (len(cov_labels)-1)
        fig = plot_covariance(
            y0, cov_list,
            cov_labels=cov_labels,
            dim_labels=dim_labels,
            linestyles=linestyles
        )
        print('Saving figure ...')
        fig.savefig(f'plots/cov_components_{n:02d}.svg')
        plt.close(fig)
        print('   -> done')
        continue
        
        #cov_tmp = io_dict['cov_comp']['dA/dtheta'][i]
        #rho_tmp = get_corr_matrix(cov_tmp)
        #print(np.array2string(
        #    rho_tmp[:6,:6],
        #    formatter={'float_kind':lambda z:'{: >7.4f}'.format(z)}
        #))
        
        #sigma_theta = np.sqrt(np.diag(d['atm_param_cov'][i]))
        corr_theta = get_corr_matrix(d['atm_param_cov'][i])
        sigma_theta = np.diag(corr_theta)
        
        title = {
            'r': (
                  f'$r = {r0[i]:.2f}$' + ' \n '
                + f'$\sigma_r = {sigma_r[i]:.2f}$'
            ),
            'dM/dtheta': (
                  r'$\mathrm{d}M/\mathrm{d}\theta$' + ' \n '
                + r'$\theta = ('
                + ', '.join([f'{x:.2f}' for x in io_dict['x'][i]])
                + r')$' + ' \n '
                + r'$\sigma_{\theta} = ('
                + ', '.join([f'{x:.2f}' for x in sigma_theta])
                + ')$' + ' \n '
                + r'$\rho_{01} = ' + f'{corr_theta[0,1]:+.3f}' + r',\ '
                + r'\rho_{02} = ' + f'{corr_theta[0,2]:+.3f}' + r',\ '
                + r'\rho_{12} = ' + f'{corr_theta[1,2]:+.3f}$'
                #+ r'C_{\theta} = '
                #+ latexify_matrix(d['atm_param_cov'][i])
            ),
            'delta_m': '$\delta m$',
            'dA/dtheta': (
                  r'$\mathrm{d}A/\mathrm{d}\theta$' + ' \n '
                + r'$\theta = ('
                + ', '.join([f'{x:.2f}' for x in io_dict['x'][i]])
                + r')$' + ' \n '
                + r'$\sigma_{\theta} = ('
                + ', '.join([f'{x:.2f}' for x in sigma_theta])
                + ')$' + ' \n '
                + r'$\rho_{01} = ' + f'{corr_theta[0,1]:+.3f}' + r',\ '
                + r'\rho_{02} = ' + f'{corr_theta[0,2]:+.3f}' + r',\ '
                + r'\rho_{12} = ' + f'{corr_theta[1,2]:+.3f}$'
                #+ r'C_{\theta} = '
                #+ latexify_matrix(d['atm_param_cov'][i])
            ),
        }
        
        rchisq_txt = r'$\chi^2/\nu = {:.3f}$'.format(io_dict['rchisq'][i])
        fig = plot_cov(io_dict['cov_y'][i], mu=y0, labels=labels)
        fig.suptitle(rchisq_txt, fontsize=22)
        fig.text(0.98, 0.98, f'{n:02d}', ha='right', va='top')
        fig.savefig(f'plots/cov_combined_{n:02d}.png')
        plt.close(fig)
        
        for comp in io_dict['cov_comp']:
            if comp == 'dm':
                continue
            
            print(f'  - {comp}')
            
            cov = io_dict['cov_comp'][comp][i]
            
            fig = plot_cov(cov, mu=y0, labels=labels)
            fig.suptitle(title[comp]+' \n '+rchisq_txt, fontsize=22)
            fig.text(0.98, 0.98, f'{n:02d}', ha='right', va='top')
            comp = comp.replace(r'/','_')
            fig.savefig(f'plots/cov_{comp}_{n:02d}.png')
            plt.close(fig)
    
    return 0


if __name__ == '__main__':
    main()

