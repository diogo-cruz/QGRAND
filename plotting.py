import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter

from .statistics_ import get_uniform_at_random_statistics, p_fail_asymp

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=24)
plt.rcParams['figure.figsize'] = [10, 6]

def plot_infidelity(n,
                    k_list,
                    num_gates,
                    p,
                    max_t,
                    infidelity,
                    confidence_interval = 0.8,
                    interpolation = 'linear',
                    add_subplot_labeling = '(a)',
                    subplot_index = 0,
                    set_linear_formatter = False,
                    fig_ax = None,
                    show = False,
                    save = False):
    """Plots the infidelity vs. code rate.

    Parameters
    ----------
    n : int
        Total number of encoding qubits.
    k_list : 1D array
        1D array of different numbers of data qubits.
    num_gates : 1D array
        List for the different number of gates considered. If the list has 
        length 1, the gate number is omitted from the plots.
    p : 1D array
        Different cases for the probability of error.
    max_t : int
        Maximum weight to be plotted.
    failure_rate : 
        Data array.
    confidence_interval : float, None, default 0.8
        Whether to consider a confidence interval for the data. If ``None``, 
        the confidence interval is not shown. If a float between 0 and 1, the 
        confidence interval is shown.
    interpolation : str, default 'linear'
        Whether to interpolate the points to be shown. Default is 'linear', 
        which is equivalent to not interpolating, as it does not change the 
        plot's appearance. In general, `interpolation` is passed as the `kind`
        keyword argument in the function `scipy.interpolate.interp1d()`.
    add_subplot_labeling : str, default '(a)'
        What labeling to add to the top left corner of the plot. For no 
        labeling, use ``''``.
    set_linear_formatter : bool, list, default False
        By default (``False`` case), a log scale is used for the y axis. To 
        use a linear scale instead, pass a list with the values that should be
        shown in the axis labeling.
    fig_ax : default None
        (fig, ax) Matplotlib figure and axes to use.
    show : bool, default False
        Whether to show the plot.
    save : bool, str, default False
        Whether to save the plot. Does not save it, by default. If ``True``, 
        saves the plot and encodes a timestamp in the filename. If a specific 
        timestamp is passed, it uses that instead.

    Returns
    -------
    fig : optional
        Matplotlib figure.
    ax : optional
        Matplotlib axes.    
    """

    
    conf = (1 - confidence_interval) / 2

    code_rate = np.array(k_list)/n
    x = np.linspace(code_rate[0], code_rate[-1], 100)
    _, failure_theory, any_error_prob = get_uniform_at_random_statistics(n,x,p,max_t)

    # E_N = np.array([(3**(i+1) * comb(n, i+1)) / 2**(n*(1-x)) for i in range(max_t)])
    # B_N = np.insert(np.cumsum(E_N, axis=0)[:-1], 0, 1/ 2**(n*(1-x)), 0)
    # y_theory = np.where(E_N < 1e-3, 1, (1-np.exp(-E_N))/E_N) * np.exp(-B_N)

    # dd = np.arange(1, max_t+1)[:,None, None]
    # failure_theory = any_error_prob - np.cumsum(p**dd * (1-p)**(n-dd) * comb(n, dd) * y_theory[:,:,None], axis=0)

    shade = True if confidence_interval is not None else False

    if fig_ax is None:
        rows = 1
        columns = 1
        fig, ax = plt.subplots(rows,
                            columns,
                            sharex='col',
                            sharey=False,
                            squeeze=False,
                            figsize=(10*columns, 6*rows))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
    else:
        fig, ax = fig_ax

    # Create axes
    ax1 = ax[subplot_index, 0]
    ax2 = ax1.twinx() if p.size > 1 else None
    
    # Add subplot labeling
    trans = mtransforms.ScaledTranslation(-65/72, -25/72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, add_subplot_labeling, transform=ax1.transAxes + trans,
                        fontsize='medium', va='bottom', fontfamily='serif')

    # Plot uncoded infidelity
    ax1.plot(code_rate,
        np.linspace(any_error_prob[0], any_error_prob[0], code_rate.shape[0]),
        'k',
        label='uncoded')
    ax2.plot(code_rate,
        np.linspace(any_error_prob[1], any_error_prob[1], code_rate.shape[0]),
        'k--') if p.size > 1 else None

    line = ['.',':','--','-.','-']
    plot = [None]*max_t

    for j, n_gates in enumerate(num_gates):
        for i in range(max_t):# + [n-1]:

            y = interp1d(code_rate, 
                        np.median(infidelity[:, j, :, i, 0], axis=1),
                        kind=interpolation)(x)
            q_m = interp1d(code_rate,
                        np.quantile(infidelity[:, j, :, i, 0], conf, axis=1),
                        kind=interpolation)(x)
            q_p = interp1d(code_rate,
                    np.quantile(infidelity[:, j, :, i, 0], 1-conf, axis=1),
                    kind=interpolation)(x)
            
            label = 'up to $t={}$'.format(i+1) if i<n-1 else 'all corrected'
            
            plot[i] = ax1.plot(x,
                y,
                line[j],
                label=label,
                color=plot[i][-1].get_color() if plot[i] is not None else None
            )
            ax1.fill_between(x,
                            q_m,
                            q_p,
                            alpha=0.2,
                            color=plot[i][-1].get_color()) if shade else None
            ax1.plot(x[::5],
                    failure_theory[i,::5,0],
                    'o',
                    markerfacecolor='none',
                    color=plot[i][-1].get_color()) if j==0 else None

            if p.size > 1:

                y = interp1d(code_rate,
                            np.median(infidelity[:, j, :, i, 1], axis=1),
                            kind=interpolation)(x)
                q_m = interp1d(code_rate,
                        np.quantile(infidelity[:, j, :, i, 1], conf, axis=1),
                        kind=interpolation)(x)
                q_p = interp1d(code_rate,
                    np.quantile(infidelity[:, j, :, i, 1], 1-conf, axis=1),
                    kind=interpolation)(x)
                
                label='up to $t={}$'.format(i+1) if i<n-1 else 'all corrected'
                
                ax2.plot(x, y, '--', color=plot[i][-1].get_color())
                ax2.fill_between(x,
                            q_m,
                            q_p,
                            alpha=0.2,
                            color=plot[i][-1].get_color()) if shade else None
                ax2.plot(x[::5],
                        failure_theory[i,::5,1],
                        'o',
                        markerfacecolor='none',
                        color=plot[i][-1].get_color()) if j==0 else None

    if num_gates.size > 1:
        for j, n_gates in enumerate(num_gates):
            ax1.plot(x,
                    np.zeros_like(x)+any_error_prob[0],
                    'k'+line[j],
                    label='{} gates'.format(n_gates))

    ax1.plot([-0.1],
            [any_error_prob[0]],
            'ko',
            markerfacecolor='none',
            label='uniform')

    # Labeling and legend
    #ax1.set_xlabel('code rate $R$, for $n = {}$'.format(n))
    ax1.set_ylabel('BLER, $p={}$, solid'.format(p[0]))
    ax2.set_ylabel(
        'BLER, $p={}$, dashed'.format(p[1])
    ) if p.size > 1 else None
    ax1.legend(loc='lower right',
                borderpad=0.2,
                labelspacing=0.15,
                handlelength=1,
                handletextpad=0.3,
                borderaxespad=0.2)
    ax1.set_yscale('log')
    ax2.set_yscale('log') if p.size > 1 else None
    ax1.set_xlim([code_rate[0], code_rate[-1]])
    if set_linear_formatter:
        ax1.set_yticks(set_linear_formatter)
        ax1.yaxis.set_major_formatter(ScalarFormatter())
    #ax2.set_ylim([5e-3, None])
    #ax1.autoscale(axis='x', tight=True)
    #plt.show()

def plot_correctable_fraction(n,
                            k_list,
                            num_gates,
                            p,
                            max_t,
                            correctable_fraction,
                            confidence_interval = 0.8,
                            interpolation = 'linear',
                            add_subplot_labeling = '(a)',
                            subplot_index = 0,
                            set_linear_formatter = False,
                            fig_ax = None,
                            show = False,
                            save = False):
    """
    Plots the correctable fraction of error rates for different code rates and gate numbers.

    Parameters
    ----------
    n : int
        The number of qubits.
    k_list : list or array-like
        The list of logical qubits.
    num_gates : int or array-like
        The number of gates.
    p : float
        The probability of an error.
    max_t : int
        The maximum number of correction steps.
    correctable_fraction : array-like
        The correctable fraction of error rates.
    confidence_interval : float, optional, default=0.8
        The confidence interval to plot the error bars. Set to None to disable.
    interpolation : str, optional, default='linear'
        The type of interpolation to use for the error bars.
    add_subplot_labeling : str, optional, default='(a)'
        The subplot label.
    subplot_index : int, optional, default=0
        The index of the subplot.
    set_linear_formatter : bool, optional, default=False
        Whether to set the formatter for the subplot to linear.
    fig_ax : tuple, optional, default=None
        The figure and axes to use for the plot. If None, a new figure will be created.
    show : bool, optional, default=False
        Whether to show the plot.
    save : bool, optional, default=False
        Whether to save the plot.

    Returns
    -------
    None

    Notes
    -----
    This function creates a plot of the correctable fraction of error rates with respect to code rate
    and number of gates. It also allows the user to specify the confidence interval and interpolation
    method for the error bars, subplot labels and index, formatter, and figure and axes for the plot.
    The plot can be displayed or saved as specified by the user.
    """

    conf = (1 - confidence_interval) / 2

    code_rate = np.array(k_list)/n
    x = np.linspace(code_rate[0], code_rate[-1], 100)
    y_theory, _, _ = get_uniform_at_random_statistics(n,x,p,max_t)

    if fig_ax is None:
        rows = 1
        columns = 1
        fig, ax = plt.subplots(rows,
                            columns,
                            sharex='col',
                            sharey=False,
                            squeeze=False,
                            figsize=(10*columns, 6*rows))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.02)
    else:
        fig, ax = fig_ax

    ax3 = ax[subplot_index,0]

    # Add subplot labeling
    trans = mtransforms.ScaledTranslation(-65/72, -25/72, fig.dpi_scale_trans)
    ax3.text(0.0, 1.0, add_subplot_labeling, transform=ax3.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')

    line = ['.',':','--','-.','-']
    plot = [None]*max_t
    shade = True if confidence_interval is not None else False

    for j, n_gates in enumerate(num_gates):
        for i in range(max_t):
            if np.mean(correctable_fraction[0, j, :, i]) < 1e-3:
                break
            y = interp1d(code_rate, np.median(correctable_fraction[:, j, :, i], axis=1), kind=interpolation)(x)
            q_m = interp1d(code_rate, np.quantile(correctable_fraction[:, j, :, i], conf, axis=1), kind=interpolation)(x)
            q_p = interp1d(code_rate, np.quantile(correctable_fraction[:, j, :, i], 1-conf, axis=1), kind=interpolation)(x)
            plot[i] = ax3.plot(x, y, line[j], label='$t={}$'.format(i+1) if j==4 else None, color=plot[i][-1].get_color() if plot[i] is not None else None)
            ax3.fill_between(x, q_m, q_p, alpha=0.2, color=plot[i][-1].get_color()) if shade else None
            ax3.plot(x, y_theory[i], 'o', markerfacecolor='none', color=plot[i][-1].get_color()) if j==1 else None

    if num_gates.size > 1:
        for j, n_gates in enumerate(num_gates):
            ax3.plot(x,
                    np.zeros_like(x)-0.1,
                    'k'+line[j],
                    label='{} gates'.format(n_gates))
    ax3.plot(x, x**0+0.1, 'ko', markerfacecolor='none', label='uniform')
    ax3.set_ylabel('$f$')
    ax3.set_xlabel('$R$')
    ax3.legend(loc='upper right',borderpad=0.2, labelspacing=0.15, handlelength=1, handletextpad=0.3, borderaxespad=0.2)
    ax3.set_ylim([0, 1])
    ax3.autoscale(axis='x', tight=True)

def plot_p_threshold(error_rate,
                    success_data,
                    success_data_all,
                    n_cx,
                    n_list,
                    filetime,
                    s=2,
                    max_t=1,
                    mode='standard',
                    name='QRLC',
                    save=False):
    """
    Plots the success probability of a quantum error correction code as a function of the error rate for different parameters.

    Parameters
    ----------
    error_rate : array_like
        Array of error rates for which the success probabilities are computed.
    success_data : array_like
        Array containing the success probabilities for the different error rates and parameters.
    success_data_all : array_like
        Array containing the success probabilities for all data points, including non-ideal ones.
    n_cx : array_like
        Array containing the number of CNOT gates for the different parameters.
    n_list : array_like
        List of block sizes for which the success probabilities are computed.
    filetime : str
        Timestamp used for saving the plot as a file.
    s : int, optional, default: 2
        The parameter for the code distance. If s=0, the plot will display the results for all distances.
    max_t : int, optional, default: 1
        Maximum number of time steps for the simulation.
    mode : str, optional, default: 'standard'
        The plotting mode. Can be either 'standard' or 'fraction'. In 'standard' mode, the y-axis shows the failure
        probability. In 'fraction' mode, the y-axis shows the ratio of the coded and uncoded success probabilities.
    name : str, optional, default: 'QRLC'
        A name to be used for the plot and the saved file.
    save : bool, optional, default: False
        If True, the plot will be saved as a PDF file.

    Returns
    -------
    None
    """

    if s==0:
        st = 0
    else:
        st = s-1
    conf = 0.2
    runs = success_data.shape[2]
    fig = plt.figure()
    if mode == 'fraction':
        if runs==1:
            for l, n in enumerate(n_list):
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                plot = plt.plot(error_rate, success_data_all[l,:,0,st]/(1-(1-error_rate)**n_cx[l,0,st]), label=label)
                plt.plot(error_rate, success_data[l,:,0,st]/(1-(1-error_rate)**n_cx[l,0,st]), ':', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                q_m = np.quantile(success_data_all[l,:,:,st], conf, axis=1)/(1-(1-error_rate)**n_cx[l,0,st])
                q_p = np.quantile(success_data_all[l,:,:,st], 1-conf, axis=1)/(1-(1-error_rate)**n_cx[l,0,st])
                plot = plt.plot(error_rate, (np.median(success_data_all[l,:,:,st], axis=1))/(1-(1-error_rate)**n_cx[l,0,st]), label=label)
                plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
                plt.plot(error_rate, (np.median(success_data[l,:,:,st], axis=1))/(1-(1-error_rate)**n_cx[l,0,st]), ':', color=plot[-1].get_color())
        plt.ylabel(r"$P_{\rm coded}/P_{\rm uncoded}$")
    elif mode == 'standard':
        if runs==1:
            for l, n in enumerate(n_list):
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                plot = plt.plot(error_rate, success_data_all[l,:,0,st], label=label)
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
                plt.plot(error_rate, success_data[l,:,0,st], ':', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                q_m = np.quantile(success_data_all[l,:,:,st], conf, axis=1)
                q_p = np.quantile(success_data_all[l,:,:,st], 1-conf, axis=1)
                plot = plt.plot(error_rate, np.median(success_data_all[l,:,:,st], axis=1), label=label)
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
                plt.plot(error_rate, np.median(success_data[l,:,:,st], axis=1), ':', color=plot[-1].get_color())
                plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
        plt.ylabel(r"$P_{\rm failure}$")
        plt.yscale('log')
    plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_{}_k1_tmax{}_s{}_{}.pdf'.format(mode,max_t,s,name)
    if save:
        fig.savefig(filename, bbox_inches='tight')

def plot_p_threshold_trans(error_rate,
                    success_data_all,
                    n_cx,
                    qe,
                    n_list,
                    filetime,
                    s=2,
                    max_t=1,
                    name='QRLC',
                    save=False):
    """
    Plot the failure probability as a function of the CNOT error rate for a given set of parameters.

    Parameters
    ----------
    error_rate : ndarray
        Array of CNOT error rates to be plotted.
    success_data_all : ndarray
        Array containing the success data for all simulations.
    n_cx : ndarray
        Array containing the number of CNOT gates for each value of n.
    qe : float
        Quantum efficiency.
    n_list : list
        List of values of n for which the simulations were performed.
    filetime : str
        Timestamp for file naming.
    s : int, optional, default=2
        The number of qubits being used for syndrome measurement (default is 2).
    max_t : int, optional, default=1
        The maximum number of time steps in the simulation.
    name : str, optional, default='QRLC'
        The name of the protocol (default is 'QRLC').
    save : bool, optional, default=False
        Whether to save the plot as a file or not (default is False).

    Returns
    -------
    None

    Notes
    -----
    This function plots the failure probability (y-axis) against the CNOT error rate (x-axis) in a log-log scale
    for each value of n in the n_list. The data points are displayed as lines, and the theoretical
    predictions are plotted as dashed lines. If there are multiple runs, the confidence intervals are
    plotted as shaded areas.

    The generated plot can be optionally saved as a PDF file in the 'Plots' directory.
    """
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(
                                    np.linspace(0,1,error_rate.shape[0],endpoint=True)))

    if s==0:
        st = 0
    else:
        st = s-1
    conf = 0.1
    runs = success_data_all.shape[3]
    fig = plt.figure()
    if runs==1:
        for l, n in enumerate(n_list):
            for j, pe in enumerate(error_rate): 
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                plot = plt.plot(error_rate, success_data_all[l,j,:,0,st], label=label if j==0 else None)
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
    else:
        for l, n in enumerate(n_list):
            for j, pe in enumerate(error_rate):
                label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                q_m = np.quantile(success_data_all[l,j,:,:,st], conf, axis=1)
                q_p = np.quantile(success_data_all[l,j,:,:,st], 1-conf, axis=1)
                plot = plt.plot(error_rate, np.median(success_data_all[l,j,:,:,st], axis=1), label=label if j==0 else None)
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
                plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
    plt.ylabel(r"$P_{\rm failure}$")
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_k1_tmax{}_s{}_{}.pdf'.format(max_t,s,name)
    if save:
        fig.savefig(filename, bbox_inches='tight')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

def plot_p_threshold_trans_diogoit(error_rate,
                    error_rate_e,
                    success_data_all,
                    n_cx,
                    qe,
                    n_list,
                    filetime,
                    s=2,
                    max_t=1,
                    max_te=1,
                    k_list = None,
                    name='QRLC',
                    save=False):
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(
                                    np.linspace(0,0.95,k_list.shape[0],endpoint=True)))

    if s==0:
        st = 0
    else:
        st = s-1
    conf = 0.1
    runs = success_data_all.shape[-2]
    fig = plt.figure()
    if len(success_data_all.shape)==6:
        if runs==1:
            for j, pe in enumerate(error_rate_e): 
                for l, n in enumerate(n_list):
                    for kk,k in enumerate(k_list):
                        label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                        plot = plt.plot(error_rate, success_data_all[kk,l,j,:,0,st], label=label if j==0 else None)
                        plt.plot(error_rate, 1-(1-error_rate)**n_cx[kk,l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
        else:
            for j, pe in enumerate(error_rate_e):
                for l, n in enumerate(n_list):
                    for kk, k in enumerate(k_list):
                        label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                        q_m = np.quantile(success_data_all[kk,l,j,:,:,st], conf, axis=1)
                        q_p = np.quantile(success_data_all[kk,l,j,:,:,st], 1-conf, axis=1)
                        plot = plt.plot(error_rate, np.median(success_data_all[kk,l,j,:,:,st], axis=1), label=label if j==0 else None)
                        plt.plot(error_rate, 1-(1-error_rate)**n_cx[kk,l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
                        plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
    else:
        if runs==1:
            for l, n in enumerate(n_list):
                for j, pe in enumerate(error_rate_e): 
                    label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    plot = plt.plot(error_rate, success_data_all[l,j,:,0,st], label=label if j==0 else None)
                    plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                for j, pe in enumerate(error_rate_e):
                    label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    q_m = np.quantile(success_data_all[l,j,:,:,st], conf, axis=1)
                    q_p = np.quantile(success_data_all[l,j,:,:,st], 1-conf, axis=1)
                    plot = plt.plot(error_rate, np.median(success_data_all[l,j,:,:,st], axis=1), label=label if j==0 else None)
                    plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
                    plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
    plt.ylabel(r"$P_{\rm failure}$")
    plt.yscale('log')
    #plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_k1_tmax{}_s{}_{}.pdf'.format(max_t,s,name)
    if save:
        fig.savefig(filename, bbox_inches='tight')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    
def plot_p_threshold_perfect(error_rate_e,
                    success_data_all,
                    n_cx,
                    qe,
                    n_list,
                    filetime,
                    s=2,
                    max_t=1,
                    max_te=1,
                    k_list = None,
                    name='QRLC',
                    save=False):
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(
                                    np.linspace(0,0.95,k_list.shape[0],endpoint=True)))

    if s==0:
        st = 0
    else:
        st = s-1
    conf = 0.1
    runs = success_data_all.shape[-2]
    fig = plt.figure()
    if len(success_data_all.shape)==6:
        if runs==1:
            for l, n in enumerate(n_list):
                for kk,k in enumerate(k_list):
                    #label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    plot = plt.plot(error_rate_e, success_data_all[kk,l,:,0,0,st])
                    plt.plot(error_rate_e, 1-(1-error_rate_e)**n, '--', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                for kk, k in enumerate(k_list):
                    #label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    q_m = np.quantile(success_data_all[kk,l,:,0,:,st], conf, axis=1)
                    q_p = np.quantile(success_data_all[kk,l,:,0,:,st], 1-conf, axis=1)
                    plot = plt.plot(error_rate_e, np.median(success_data_all[kk,l,:,0,:,st], axis=1))
                    plt.plot(error_rate_e, 1-(1-error_rate_e)**n, '--', color=plot[-1].get_color())
                    plt.fill_between(error_rate_e, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
    else:
        if runs==1:
            for l, n in enumerate(n_list):
                for j, pe in enumerate(error_rate_e): 
                    label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    plot = plt.plot(error_rate, success_data_all[l,j,:,0,st], label=label if j==0 else None)
                    plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                for j, pe in enumerate(error_rate_e):
                    label = "$n={}, s={}$".format(n, st+1) if s>0 else "$n={}$".format(n)
                    q_m = np.quantile(success_data_all[l,j,:,:,st], conf, axis=1)
                    q_p = np.quantile(success_data_all[l,j,:,:,st], 1-conf, axis=1)
                    plot = plt.plot(error_rate, np.median(success_data_all[l,j,:,:,st], axis=1), label=label if j==0 else None)
                    plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st] * (1-pe)**(qe*n), '--', color=plot[-1].get_color())
                    plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
                    
    A = lambda t,n: 3**t * comb(n, t)
    B = lambda t,n: sum(A(i,n) for i in range(t+1))

    S = lambda n,k: 2**(n-k)
    f_corr = lambda t,n,a: (S(n)/A(t,n,a)) * np.exp(-B(t-1,n,a)/S(n)) * (1 - np.exp(-A(t,n,a)/S(n)))
    f_corr2 = lambda t,n,k: ((S(n,k)/A(t,n)) * np.exp(B(t-1,n)*np.log(1-1/S(n,k))+np.log(1-np.exp(A(t,n)*np.log(1-1/S(n,k))))))

    def Pa(p,n,k):
        return 1-sum(f_corr2(t,n,k)*comb(n, t) * np.exp(t*np.log(p) + (n-t)*np.log(1-p)) for t in range(n+1))#, 1-(1-p)**Nc
    #plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    color = ['r','g','b','k']
    ks = [4,6,8,10]
    for i,n in enumerate(n_list):
        plt.plot(error_rate_e, (1-1/4**[ks[i]])*Pa(error_rate_e,n,ks[i]), color=color[i])
    plt.ylabel(r"$P_{\rm failure}$")
    plt.yscale('log')
    #plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_k1_tmax{}_s{}_{}.pdf'.format(max_t,s,name)
    if save:
        fig.savefig(filename, bbox_inches='tight')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)


def plot_p_threshold_asymp(error_rate,
                    success_data,
                    success_data_all,
                    n_cx,
                    n_list,
                    filetime,
                    s=2,
                    max_t=1,
                    mode='standard',
                    name='QRLC',
                    D=0.3,
                    save=False):
    """
    Plot the performance of a quantum error correction code in terms of error rate and success probabilities.

    Parameters
    ----------
    error_rate : array_like
        An array of CNOT error rates.
    success_data : array_like
        Success data for the quantum error correction code.
    success_data_all : array_like
        Additional success data for the quantum error correction code.
    n_cx : array_like
        The number of CNOT gates for each lattice size and sweep number.
    n_list : list
        List of lattice sizes.
    filetime : str
        Time identifier for output file name.
    s : int, optional, default: 2
        Sweep number.
    max_t : int, optional, default: 1
        Maximum number of time steps.
    mode : {'standard', 'fraction'}, optional, default: 'standard'
        Plot mode. 'standard' plots the probability of failure, while 'fraction' plots the ratio of coded to uncoded success probabilities.
    name : str, optional, default: 'QRLC'
        Short name for the quantum error correction code.
    D : float, optional, default: 0.3
        Decoder parameter.
    save : bool, optional, default: False
        If True, saves the plot to a file.

    Returns
    -------
    None

    Notes
    -----
    This function generates a plot of the performance of a quantum error correction code in terms of error rate and success probabilities.
    In 'standard' mode, it plots the probability of failure for the given error rates and lattice sizes. In 'fraction' mode, it plots the
    ratio of coded to uncoded success probabilities for the given error rates and lattice sizes.
    """

    st = s-1
    conf = 0.1
    runs = success_data.shape[2]
    fig = plt.figure()
    if mode == 'fraction':
        if runs==1:
            for l, n in enumerate(n_list):
                plot = plt.plot(error_rate, success_data_all[l,:,0,st]/(1-(1-error_rate)**n_cx[l,0,st]), label="$n={}, s={}$".format(n**2 + (n-1)**2, st+1))
                plt.plot(error_rate, success_data[l,:,0,st]/(1-(1-error_rate)**n_cx[l,0,st]), ':', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                q_m = np.quantile(success_data_all[l,:,:,st], conf, axis=1)/(1-(1-error_rate)**n_cx[l,0,st])
                q_p = np.quantile(success_data_all[l,:,:,st], 1-conf, axis=1)/(1-(1-error_rate)**n_cx[l,0,st])
                plot = plt.plot(error_rate, (np.median(success_data_all[l,:,:,st], axis=1))/(1-(1-error_rate)**n_cx[l,0,st]), label="$n={}, s={}$".format(n, st+1))
                plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
                plt.plot(error_rate, (np.median(success_data[l,:,:,st], axis=1))/(1-(1-error_rate)**n_cx[l,0,st]), ':', color=plot[-1].get_color())
        plt.ylabel(r"$P_{\rm coded}/P_{\rm uncoded}$")
    elif mode == 'standard':
        if runs==1:
            for l, n in enumerate(n_list):
                plot = plt.plot(error_rate, success_data_all[l,:,0,st], label="$n={}, s={}$".format(n**2 + (n-1)**2, st+1))
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
                plt.plot(error_rate, success_data[l,:,0,st], ':', color=plot[-1].get_color())
                plt.plot(error_rate, p_fail_asymp(error_rate, n_cx[l,0,st], n, D=D), '-.', color=plot[-1].get_color())
        else:
            for l, n in enumerate(n_list):
                q_m = np.quantile(success_data_all[l,:,:,st], conf, axis=1)
                q_p = np.quantile(success_data_all[l,:,:,st], 1-conf, axis=1)
                plot = plt.plot(error_rate, np.median(success_data_all[l,:,:,st], axis=1), label="$n={}, s={}$".format(n, st+1))
                plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
                plt.plot(error_rate, np.median(success_data[l,:,:,st], axis=1), ':', color=plot[-1].get_color())
                plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
                plt.plot(error_rate, p_fail_asymp(error_rate, n_cx[l,0,st], n, D=D), '-.', color=plot[-1].get_color())
        plt.ylabel(r"$P_{\rm failure}$")
        plt.yscale('log')
    plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_{}_k1_tmax{}_s{}_{}.pdf'.format(mode,max_t,s,name)
    if save:
        fig.savefig(filename, bbox_inches='tight')

def plot_p_threshold_s(error_rate,
                    success_data,
                    success_data_all,
                    n_cx,
                    n_list,
                    filetime,
                    s,
                    max_t=1,
                    mode='standard',
                    name='QRLC',
                    save=False):
    """
    Plots the probability of failure versus CNOT error rate for different lattice sizes and stabilizer measurements.

    Parameters
    ----------
    error_rate : array_like
        Array of CNOT error rates.
    success_data : ndarray
        Array containing success data for each lattice size, error rate, and number of stabilizer measurements.
    success_data_all : ndarray
        Array containing success data for all runs.
    n_cx : ndarray
        Array containing the number of CNOT gates for each lattice size and number of stabilizer measurements.
    n_list : list
        List of lattice sizes.
    filetime : str
        Timestamp string used for saving the plot.
    s : list
        List of stabilizer measurements.
    max_t : int, optional
        Maximum time for stabilizer measurements, default is 1.
    mode : str, optional
        Mode for plotting data, default is 'standard'.
    name : str, optional
        Name of the quantum error correction code, default is 'QRLC'.
    save : bool, optional
        If True, saves the plot as a PDF file, default is False.

    Returns
    -------
    None
    """

    #st = s-1
    conf = 0.1
    runs = success_data.shape[2]
    fig = plt.figure()
    if mode == 'standard':
        if runs==1:
            for l, n in enumerate(n_list):
                for st, stabs in enumerate(s):
                    plot = plt.plot(error_rate, success_data_all[l,:,0,st], label="$n={}, s={}$".format(n**2 + (n-1)**2, stabs))
                    plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
                    plt.plot(error_rate, success_data[l,:,0,st], ':', color=plot[-1].get_color())
        # else:
        #     for l, n in enumerate(n_list):
        #         q_m = np.quantile(success_data_all[l,:,:,st], conf, axis=1)
        #         q_p = np.quantile(success_data_all[l,:,:,st], 1-conf, axis=1)
        #         plot = plt.plot(error_rate, np.median(success_data_all[l,:,:,st], axis=1), label="$n={}, s={}$".format(n, st+1))
        #         plt.plot(error_rate, 1-(1-error_rate)**n_cx[l,0,st], '--', color=plot[-1].get_color())
        #         plt.plot(error_rate, np.median(success_data[l,:,:,st], axis=1), ':', color=plot[-1].get_color())
        #         plt.fill_between(error_rate, q_m, q_p, alpha=0.2, color=plot[-1].get_color())
        plt.ylabel(r"$P_{\rm failure}$")
        plt.yscale('log')
    plt.legend()
    plt.xlabel(r"$p_{\rm CNOT}$")
    plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    filename = 'Plots/'+filetime+'_ft_p_threshold_{}_k1_tmax{}_s{}_{}.pdf'.format(mode,max_t,s[-1],name)
    if save:
        fig.savefig(filename, bbox_inches='tight')