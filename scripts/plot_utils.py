#!/usr/bin/env python
#
# plot_utils.py
# Useful plotting functions.
#
# Copyright (C) 2016  Gregory M. Green
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from __future__ import print_function

import numpy as np
import matplotlib.ticker as ticker


def unit_formatter(base_unit, fmt='{:g} {:s}', prefixes=None, sigfigs=None):
    if prefixes is None:
        prefixes = ['', 'k', 'M', 'G', 'T']
    
    def f(x):
        order = int(np.floor(np.log10(np.abs(x)) / 3))
        z = x * 10.**(-3.*order)
        if sigfigs is not None:
            z = float(('{:.'+str(sigfigs)+'g}').format(z))
        return fmt.format(z, prefixes[order] + base_unit)
    
    return f


def rel2abs_ax_pos(ax, x, y):
    lim = (ax.get_xlim(), ax.get_ylim())
    return (l[0] + z * (l[1]-l[0]) for z,l in zip((x,y),lim))


def pm_f_formatter(n_digits=2):
    fmt_str = r'${{:+.{:d}f}}$'.format(n_digits)
    zero_str = r'$+0.' + n_digits * '0' + r'$'
    zero_str_correct = r'$\pm 0.' + n_digits * '0' + r'$'

    def x2str(x, pos):
        txt = fmt_str.format(x)
        if txt == zero_str:
            return zero_str_correct
        return txt

    return ticker.FuncFormatter(x2str)


def pm_deg_formatter(pad_negative=False):
    """
    Returns a function that formats angles with a "+-"-sign and a degree
    symbol.
    """
    fmt_str = r'${:+g}^{{\circ}}$'
    zero_str = r'$+0^{\circ}$'
    zero_str_correct = r'$\pm 0^{\circ}$'

    def f(theta):
        txt = fmt_str.format(theta)
        if txt == zero_str:
            return zero_str_correct
        if pad_negative and txt.startswith(r'$-'):
            txt = txt[:-1] + r'\ $'
        return txt

    return f


def percentile_weighted(x, w, pct):
    idx = np.argsort(x)
    w_cumulative = 100. * np.cumsum(w[idx_sort]) / np.sum(w)
    idx = np.searchsorted(w_cumulative, pct)
    idx = np.clip(idx, 0, x.size-1)
    return x[idx]


def correlation_plot(ax, x, y,
                     x_bins=25,
                     y_bins=20,
                     ymax_pct=99.,
                     ymax_abs=None,
                     weights=None,
                     norm='equal_ink',
                     pct=[15.87, 50., 84.13],
                     ax_hist=None,
                     fontsize=10.):
    """
    Generates a Hogg-style correlation plot between variables ``x`` and ``y`` on
    the given axes.

    Args:
        ax (``matplotlib.axes._axes.Axes``): Axes on which to draw correlation
            plot.
        x (``numpy.ndarray``): Values to put on the x-axis.
        y (``numpy.ndarray``): Values to put on the y-axis. Typically the
            difference between two variables to be compared.
        x_bins (Optional[int or float array]): Number of bins along the x-axis.
            Defaults to ``25``. If an array is provided, then it is interpreted
            as the bin edges.
        y_bins (Optional[int or float array]): Number of bins along the y-axis.
            Defaults to ``20``. If an array is provided, then it is interpreted
            as the bin edges. If ``y_bins`` is an array, it overrides both
            ``ymax_pct`` and ``ymax_abs``.
        ymax_pct (Optional[float]): The y-axis will extend to this percentile of
            the absolute values of ``y``. Defaults to ``99``.
        ymax_abs (Optional[float]): The y-axis limits will be set to this value.
            If ``None`` (the default), the ``ymax_pct`` will be used instead to
            set the y-axis limits.
        weights (Optional[float array]): Statistical weight of each value in the
            ``x`` and ``y`` arrays. Defaults to ``None`` (i.e., equal weights).
        norm (Optional[str]): The method to use to normalize the densities in
            the correlation plot. There are three possible values:
            ``'equal_ink'`` (the default) uses an equal amount of ink for each
            x-bin; ``'max'`` normalizes each x-bin separately to the maximum
            value in the bin; ``'total_weight'`` simply represents the total
            statistical weight in each xy-bin.
        pct (Optional[length-3 array]): The percentiles to show on the
            correlation plot. Defaults to ``[15.87, 50., 84.13]``, corresponding
            to the median and 1-sigma-equivalent percentiles (for a Gaussian).
        ax_hist (Optional[``matplotlib.axes._axes.Axes``]): Axes on which to
            plot a histogram of ``y``. Ideally, this axis should be to the right
            or left of ``ax``. Defaults to ``None``, meaning that no histogram
            is plotted.
        fontsize (Optional[float]): Base fontsize of labels. Some labels (e.g.,
            histogram labels) will have slightly smaller fontsizes. Defaults to
            10.

    Raises:
        ``ValueError``: ``pct`` or ``norm`` are not correctly specified.
    """

    # Determine limits on y
    if ymax_abs is None:
        y_max = 1.1 * np.percentile(np.abs(y), ymax_pct)
    else:
        y_max = ymax_abs

    # x- and y-bin edges
    if not hasattr(x_bins, '__len__'):
        x_bins = np.linspace(np.min(x), np.max(x), x_bins+1)

    if not hasattr(y_bins, '__len__'):
        y_bins = np.linspace(-y_max, y_max, y_bins+1)

    n_bins = (len(x_bins)-1, len(y_bins)-1)

    # Set up empty density map and percentile thresholds
    density = np.zeros(n_bins, dtype='f4')
    thresholds = np.zeros((n_bins[0], 3), dtype='f4')

    # Percentiles to mark
    if not hasattr(pct, '__len__'):
        raise ValueError('`pct` must have length 3.')
        if len(pct) != 3:
            raise ValueError('`pct` must have length 3.')

    # Density histogram and percentile thresholds in each x bin
    for n,(x0,x1) in enumerate(zip(x_bins[:-1], x_bins[1:])):
        idx = (x >= x0) & (x < x1)

        if np.any(idx):
            y_sel = y[idx]

            if weights is None:
                thresholds[n,:] = np.percentile(y_sel, pct)
                w = None
            else:
                w = weights[idx]
                thresholds[n,:] = np.percentile(y_sel, w, pct)

            density[n,:], _ = np.histogram(
                y_sel,
                bins=y_bins,
                density=False,
                range=[-y_max, y_max],
                weights=w)

    # Normalize density map
    if norm == 'max':
        a = np.max(density, axis=1)
        a[a == 0] = 1
        density /= a[:,None]
        #density[~np.isfinite(density)] = 0.
    elif norm == 'equal_ink':
        a = np.sum(density, axis=1)
        a[a == 0] = 1
        density /= a[:,None]
        n_occupied = np.count_nonzero(density > 0, axis=1)
        idx = (n_occupied > 3)
        print(f'{np.count_nonzero(~idx)} of {idx.size} ruled out.')
        # density[~np.isfinite(density)] = 0.
        norm = np.nanpercentile(np.nanmax(density, axis=1)[idx], 90.)
        print(np.max(density, axis=1))
        print(norm)
        if norm == 0:
            norm = np.max(density)
        norm = np.clip(norm, 0., 0.25)
        print(norm)
        print('')
        density /= norm
    elif norm == 'total_weight':
        density /= np.max(density)
    else:
        raise ValueError("`norm` must be 'max', 'equal_ink' or 'total_weight'.")

    # Fix NaN densities
    idx = ~np.isfinite(density)
    density[idx] = 0.

    # Plot density in background
    extent = (x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])

    ax.imshow(
        density.T,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='gray_r',
        interpolation='nearest',
        vmin=0.,
        vmax=1.)

    # Line at y = 0
    ax.axhline(y=0., c='c', ls='-', alpha=0.3)

    # Histogram of y-values
    if ax_hist is not None:
        # ax_hist.hist(
        #     y,
        #     bins=y_bins,
        #     weights=weights,
        #     orientation='horizontal',
        #     color='k',
        #     alpha=0.25)

        bin_val, bin_edges = np.histogram(y, bins=y_bins, weights=weights)
        # ax_hist.barh(bin_edges, bin_val, align='edge')

        if weights is None:
            y_pct = np.percentile(y, pct)
        else:
            y_pct = percentile_weighted(y, weights, pct)

        y_pct_padded = np.hstack([-np.inf, y_pct, np.inf])
        hist_alpha = [0.25, 0.5, 0.5, 0.25]

        for k, (y0, y1) in enumerate(zip(y_pct_padded[:-1], y_pct_padded[1:])):
            idx0, idx1 = np.searchsorted(bin_edges, [y0, y1])

            section_edges = np.hstack([y0, bin_edges[idx0:idx1], y1])

            w_idx0, w_idx1 = idx0-1, idx1

            if section_edges[0] < bin_edges[0]:
                section_edges = section_edges[1:]
                w_idx0 += 1
            if section_edges[-1] > bin_edges[-1]:
                section_edges = section_edges[:-1]
                w_idx1 -= 1

            w_section = bin_val[w_idx0:w_idx1]

            # print('(y0, y1) = ({:.2f}, {:.2f})'.format(y0, y1))
            # print('(i0, i1) = ({:d}, {:d})'.format(idx0, idx1))
            # print('(w0, w1) = ({:d}, {:d})'.format(w_idx0, w_idx1))

            # idx_bin_keep = (bin_edges > y0) & (bin_edges < y1)
            # section_edges = hp.hstack([bin_edges])

            # section_edges = bin_edges[:]
            # w_section = bin_val
            y_section = 0.5 * (section_edges[1:] + section_edges[:-1])

            ax_hist.hist(
                y_section,
                bins=section_edges,
                weights=w_section,
                orientation='horizontal',
                histtype='stepfilled',
                color='k',
                edgecolor='k',
                alpha=hist_alpha[k])

        # for yy in y_pct:
        #     ax_hist.axhline(y=yy, c='b', ls='-', alpha=0.5)

        ax_hist.yaxis.tick_right()
        ax_hist.tick_params(axis='y', which='both', direction='out')
        # ax_hist.yaxis.set_major_formatter(pm_f_formatter(n_digits=2))
        # ax_hist.yaxis.set_ticks([])
        ax_hist.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_hist.yaxis.set_ticklabels([])
        # ticker.FormatStrFormatter('$%.3f$'))

        ax_hist.set_ylim(extent[2:])
        ax_hist.set_xticks([])
        ax_hist.axis('off')
        # ax_hist.set_yticks(y_pct)
        # ax_hist.tick_params(axis='y', labelsize=0.8*fontsize)

    # Envelope of percentile thresholds
    x_range = np.linspace(x_bins[0], x_bins[-1], n_bins[0]+1)
    for i in range(3):
        y_envelope = np.hstack([[thresholds[0,i]], thresholds[:,i]])
        ax.step(x_range, y_envelope, where='pre', c='b', alpha=0.5, lw=1.0)

    # Set axis limits
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

    # Set axis ticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.tick_params(axis='y', right=True, labelright=False)
