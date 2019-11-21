import numpy as np
import scipy.ndimage
import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_chain(x,name=None,filename=None):
    """
    Produce a trace plot
    """
    fig=plt.figure(figsize=(4,3))
    plt.plot(x,',')
    plt.grid()
    plt.xlabel('iteration')
    if name is not None:
        plt.ylabel(name)
        if filename is None:
            filename=name+'_chain.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

def plot_hist(x,name=None,filename=None):
    """
    Produce a histogram
    """
    fig=plt.figure(figsize=(4,3))
    plt.hist(x, density = True, facecolor = '0.5', bins=int(len(x)/20))
    plt.ylabel('probability density')
    if name is not None:
        plt.xlabel(name)
        if filename is None:
            filename=name+'_hist.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()

def plot_corner(xs,filename=None,**kwargs):
    """
    Produce a corner plot
    """
    import corner
    fig=plt.figure(figsize=(10,10))
    mask = [i for i in range(xs.shape[-1]) if not all(xs[:,i]==xs[0,i]) ]
    corner.corner(xs[:,mask],**kwargs)
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close()


def get_contours(x, y, bins=20, levels=None, smooth=1.0, weights=None):

    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 weights=weights)
    if smooth is not None:
        H =  scipy.ndimage.gaussian_filter(H, smooth)

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m) and not quiet:
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    return X2, Y2, H2, V, H

def plot_corner_contour(x, filename=None, parameters=None, labels=None, labels_dict=None):
    """
    Make a set of contour plots
    """
    from matplotlib.lines import Line2D
    import scipy.stats as stats

    print("Making scatter plot...")
    # scatter of error on logL
    if not isinstance(x, list):
        x = [x]
        multiple_inputs = False
    if len(x) > 1:
        multiple_inputs = True

    N_params = np.shape(x[0])[-1]
    # setup colours
    marker_colours = ['tab:red', 'tab:blue']
    cm = [plt.cm.Reds, plt.cm.Blues]
    contour_colours = [c(np.linspace(0.5, 1., 3)) for c in cm]
    # setup labels
    if parameters is None:
        parameters = ["Parameter " + str(i) for i in range(N_params)]
    if labels is None:
        labels = ["Data " + str(i) for i in range(len(x))]
    # crop plots
    crop_plot = False
    if np.min(x[0]) >= 0 and np.max(x[0]) <= 1.:
        crop_plot = True
    # main loop
    contour_plots = []
    fig, axes = plt.subplots(N_params, N_params, figsize=(5*N_params, 5*N_params))
    for i in range(N_params):
        for j in range(N_params):
            ax = axes[i, j]
            ax.tick_params(axis='both', direction='in')
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            if j < i:
                idx = [j, i]
                # loop over arrays of inputs
                for n, a in enumerate(x):
                    sp = a[:, idx].T
                    levels = [0.68, 0.9, 0.95]
                    X2, Y2, H2, V, H = get_contours(*sp, levels=levels)
                    ax.contourf(X2, Y2, H2.T, np.concatenate([V, [H.max()*(1+1e-4)]]),
                            alpha=0.5, colors=contour_colours[n])
                    cp = ax.contour(X2, Y2, H2.T, V, colors=contour_colours[n], linewidths=2.0)
                    contour_plots.append(cp)
                if crop_plot:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    if xlim[0] < 0 and xlim[1] > 1:
                        ax.set_xlim([0,1])
                    elif xlim[0] < 0:
                        ax.set_xlim([0, xlim[1]])
                    elif xlim[1] > 1:
                        ax.set_ylim([xlim[0], 1])
                    if ylim[0] < 0 and ylim[1] > 1:
                        ax.set_ylim([0,1])
                    elif ylim[0] < 0:
                        ax.set_ylim([0, ylim[1]])
                    elif ylim[1] > 1:
                        ax.set_ylim([ylim[0], 1])

            elif j == i:
                h_vec = []
                for n, a in enumerate(x):
                    h = a[:, j].T
                    h_vec.append(h)
                    ax.hist(h, density=True, histtype='stepfilled',
                            color=marker_colours[n], alpha=0.5, bins=20)
                if multiple_inputs:
                    D, p_value = stats.ks_2samp(*h_vec)
                    ax.set_title("D = {:.3}, p-value= {:.4}".format(D, p_value))
                if crop_plot:
                    xlim = ax.get_xlim()
                    if xlim[0] < 0 and xlim[1] > 1:
                        ax.set_xlim([0,1])
                    elif xlim[0] < 0:
                        ax.set_xlim([0, xlim[1]])
                    elif xlim[1] > 1:
                        ax.set_ylim([xlim[0], 1])
                ax.set_yticklabels([])
            else:
                ax.set_axis_off()


            if i + 1 < N_params:
                ax.get_shared_x_axes().join(ax, axes[N_params - 1, j])
                ax.set_xticklabels([])
            else:
                if parameters is not None:
                    if labels_dict is not None:
                        ax.set_xlabel(labels_dict[parameters[j]])
                    else:
                        ax.set_xlabel(parameters[j])
            if j > 0:
                ax.set_yticklabels([])
                # histograms on diagonal (i==j) do not share y-axis
                if not i == j:
                    ax.get_shared_y_axes().join(ax, axes[i, 0])
            else:
                if not i == 0:
                    if parameters is not None:
                        if labels_dict is not None:
                            ax.set_ylabel(labels_dict[parameters[i+1]])
                        else:
                            ax.set_ylabel(parameters[i])
    # make legend
    legend_lines = [Line2D([0], [0], color=c[0], lw=2) for c in contour_colours]
    fig.legend(legend_lines, labels)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return fig
