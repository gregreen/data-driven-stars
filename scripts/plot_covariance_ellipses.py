#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec

def confidence_ellipse(ax, mu, cov, n_std=1.0, **kwargs):
    rho = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
    rho = np.clip(rho, -1., 1.)
    r_x = np.sqrt(1+rho)
    r_y = np.sqrt(1-rho)
    
    ellipse = patches.Ellipse((0.,0.), width=2*r_x, height=2*r_y, **kwargs)
    
    s_x = n_std * np.sqrt(cov[0,0])
    s_y = n_std * np.sqrt(cov[1,1])
    
    transf = transforms.Affine2D().rotate_deg(45.0).scale(s_x,s_y).translate(mu[0], mu[1])
                                                          
    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)


def plot_covariance(mu, cov_list, cov_labels=None, dim_labels=None, n_std=1.0, **kwargs):
    n_dim = cov_list[0].shape[0]
    
    # Set up figure and grid of axes
    figsize = (2.0*n_dim, 2.0*n_dim)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        n_dim-1, n_dim-1,
        figure=fig,
        wspace=0.05,
        hspace=0.05
    )
    
    # Use default color cycle
    kw = kwargs.copy()
    colors = kw.pop('colors', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyles = kw.pop('linestyles', ['-' for cov in cov_list])
    
    # Determine axis limits based on variances
    dx = np.array([
        np.max([
            1.2 * n_std * np.sqrt(cov[i,i])
            for cov in cov_list
        ])
        for i in range(n_dim)
    ])
    lim = np.vstack([mu-dx,mu+dx]).T
    
    cov_patches = [None for cov in cov_list]
    
    for row in range(n_dim-1):
        k = row+1
        
        for col in range(row+1):
            j = col
            
            ax = fig.add_subplot(gs[row,col])
            
            # Format axis labels and ticks
            if row == n_dim-2:
                if dim_labels is not None:
                    ax.set_xlabel(dim_labels[j])
            else:
                ax.set_xticklabels([])
                
            if col == 0:
                if dim_labels is not None:
                    ax.set_ylabel(dim_labels[k])
            else:
                ax.set_yticklabels([])
            
            # Set axis limits
            ax.set_xlim(lim[j])
            ax.set_ylim(lim[k])
            
            # Plot ellipse for each covariance
            for i,cov in enumerate(cov_list):
                c = cov[np.ix_([j,k], [j,k])]
                cov_patches[i] = confidence_ellipse(
                    ax, mu[[j,k]], c,
                    facecolor='none',
                    edgecolor=colors[i],
                    linestyle=linestyles[i],
                    **kw
                )
    
    # Legend
    if cov_labels is not None:
        ax = fig.add_subplot(gs[0, n_dim-2])
        ax.axis('off')
        ax.legend(cov_patches, cov_labels)
    
    return fig


def main():
    # Generate covariance matrices
    cov_list = []
    for i in range(2):
        A = np.random.normal(size=(4,2+i))
        cov_list.append(np.dot(A, A.T))
    cov_list = [np.sum(np.stack(cov_list), axis=0)] + cov_list
    
    # Make up mean for each dimension
    mu = np.array([0., 1., -1., 3.])
    
    # Labels for the dimensions and covariance components
    dim_labels = [f'ax{i}' for i in range(4)]
    cov_labels = [
        r'$\mathrm{total}$',
        r'$\left( \delta_{\theta} \cdot \nabla_{\theta} \right) \vec{M}$',
        r'$\delta \vec{m}$'
    ]
    
    # Styling for the covariance components
    linestyles=[':', '-', '-']

    # Plot the covariance components
    fig = plot_covariance(
        mu, cov_list,
        cov_labels=cov_labels,
        dim_labels=dim_labels,
        linestyles=linestyles,
        alpha=0.9
    )
    
    # Save figure
    fig.savefig('covariance_component_example.svg')
    
    return 0


if __name__ == '__main__':
    main()
