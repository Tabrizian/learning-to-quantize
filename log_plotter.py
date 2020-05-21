from scipy import interpolate
import numpy as np
import os
import re
import torch
import pylab as plt
import matplotlib.ticker as mtick
from tensorboard.backend.event_processing import event_accumulator


def get_run_names(logdir, patterns):
    run_names = []
    for pattern in patterns:
        for root, subdirs, files in os.walk(logdir, followlinks=True):
            if re.match(pattern, root):
                run_names += [root]
    # print(run_names)
    run_names.sort()
    return run_names


def get_run_names_events(logdir, patterns):
    run_names = {}
    for pattern in patterns:
        for root, subdirs, files in os.walk(logdir, followlinks=True):
            if re.match(pattern, root):
                run_names[root] = []
                for file in files:
                    if re.match('.*events\.out.*', file):
                        run_names[root].append(file)
                run_names[root] = sorted(run_names[root])
    # print(run_names)
    return run_names


def get_data_pth(logdir, run_names, tag_names, batch_size=None):
    data = []
    for run_name in run_names:
        d = {}
        logdata = torch.load(run_name + '/log.pth.tar')
        for tag_name in tag_names:
            if tag_name not in logdata:
                continue
            js = logdata[tag_name]
            d[tag_name] = np.array([[x[j] for x in js]
                                    for j in range(1, 3)])
        data += [d]
    return data


def get_data_pth_events(logdir, run_names, tag_names, batch_size=None):
    data = []
    for run_name, events in run_names.items():
        d = {}

        for event in events:
            ea = event_accumulator.EventAccumulator(run_name+'/'+event,
                                                    size_guidance={  # see below regarding this argument
                                                        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                        event_accumulator.IMAGES: 4,
                                                        event_accumulator.AUDIO: 4,
                                                        event_accumulator.SCALARS: 0,
                                                        event_accumulator.HISTOGRAMS: 1,
                                                    })
            ea.Reload()
            for tag_name in tag_names:
                if tag_name not in ea.Tags()['scalars']:
                    continue
                scalar = ea.Scalars(tag_name)
                if tag_name not in d:
                    d[tag_name] = np.array(
                        [[dp.step for dp in scalar], [dp.value for dp in scalar]])
                else:
                    new_array = np.array([dp.step for dp in scalar])
                    indexes = new_array > d[tag_name][0][-1]
                    res1 = np.concatenate(
                        (d[tag_name][0], np.array([dp.step for dp in scalar])[indexes]))
                    res2 = np.concatenate(
                        (d[tag_name][1], np.array([dp.value for dp in scalar])[indexes]))
                    d[tag_name] = (res1, res2)
        data += [d]
    return data



def plot_smooth(x, y, npts=100, order=3, *args, **kwargs):

    x_smooth = np.linspace(x.min(), x.max(), npts)
    tck = interpolate.splrep(x, y, s=0)
    y_smooth = interpolate.splev(x_smooth, tck, der=0)

    plt.plot(x_smooth, y_smooth, *args, **kwargs)


def plot_smooth_o1(x, y, *args, **kwargs):
    plot_smooth(x, y, 100, 1, *args, **kwargs)


def get_legend(lg_tags, run_name, lg_replace=[]):
    lg = ""
    for lgt in lg_tags:
        res = ".*?($|,)" if ',' not in lgt and '$' not in lgt else ''
        mg = re.search(lgt + res, run_name)
        if mg:
            lg += mg.group(0)
    lg = lg.replace('_,', ',')
    lg = lg.strip(',')
    for a, b in lg_replace:
        lg = lg.replace(a, b)
    return lg


def plot_tag(data, plot_f, run_names, tag_name, lg_tags, ylim=None, color0=0,
             ncolor=None, lg_replace=[], no_title=False):
    xlabel = {}
    ylabel = {'Tacc': 'Training Accuracy (%)', 'Terror': 'Training Error (%)',
              'train/accuracy': 'Training Accuracy (%)',
              'Vacc': 'Test Accuracy (%)', 'Verror': 'Test Error (%)',
              'valid/accuracy': 'Test Accuracy (%)',
              'loss': 'Loss',
              'epoch': 'Epoch',
              'Tloss': 'Loss', 'Vloss': 'Loss', 'lr': 'Learning rate',
              'grad_bias': 'Gradient Diff norm',
              'est_var': 'Mean variance',
              'est_snr': 'Mean SNR',
              'nb_error': 'NB Error',
              'est_nvar': 'Mean Normalized Variance'}
    titles = {'Tacc': 'Training Accuracy', 'Terror': 'Training Error',
              'train/accuracy': 'Training Accuracy',
              'Vacc': 'Test Accuracy', 'Verror': 'Test Error',
              'loss': 'Loss',
              'epoch': 'Epoch',
              'Tloss': 'Loss on full training set', 'lr': 'Learning rate',
              'Vloss': 'Loss on validation set',
              'grad_bias': 'Optimization Step Bias',
              'nb_error': 'Norm-based Variance Error',
              'est_var': 'Optimization Step Variance (w/o learning rate)',
              'est_snr': 'Optimization Step SNR',
              'est_nvar': 'Optimization Step Normalized Variance (w/o lr)',
              }
    yscale_log = ['Tloss', 'Vloss', 'est_var']  # , 'est_var'
    yscale_base = []
    # yscale_sci = ['est_bias', 'est_var']
    plot_fs = {'Tacc': plot_f, 'Vacc': plot_f,
               'Terror': plot_f, 'Verror': plot_f,
               'Tloss': plot_f, 'Vloss': plot_f,
               }
    for k in list(ylabel.keys()):
        if k not in xlabel:
            xlabel[k] = 'Training Iteration'
        if k not in plot_fs:
            plot_fs[k] = plot_f
        if k not in plot_fs:
            plot_fs[k] = plt.plot

    if not isinstance(data, list):
        data = [data]
        run_names = [run_names]

    color = ['blue', 'orangered', 'limegreen', 'darkkhaki', 'cyan', 'grey']
    color = color[:ncolor]
    style = ['-', '--', ':', '-.']
    # plt.rcParams.update({'font.size': 12})
    plt.grid(linewidth=1)
    legends = []
    for i in range(len(data)):
        if tag_name not in data[i]:
            continue
        legends += [get_legend(lg_tags, run_names[i], lg_replace)]
        plot_fs[tag_name](
            data[i][tag_name][0], data[i][tag_name][1],
            linestyle=style[(color0 + i) // len(color)],
            color=color[(color0 + i) % len(color)], linewidth=2)
    if not no_title:
        plt.title(titles[tag_name])
    if tag_name in yscale_log:
        ax = plt.gca()
        if tag_name in yscale_base:
            ax.set_yscale('log', basey=np.e)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
        else:
            ax.set_yscale('log')
    else:
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    if ylim is not None:
        plt.ylim(ylim)
    # plt.xlim([0, 25000])
    plt.legend(legends, bbox_to_anchor=(1.1, 1.05))
    plt.xlabel(xlabel[tag_name])
    plt.ylabel(ylabel[tag_name])


def ticks(y, pos):
    return r'$e^{{{:.0f}}}$'.format(np.log(y))


def plot_runs_and_tags(get_data_f, plot_f, logdir, patterns, tag_names,
                       fig_name, lg_tags, ylim, batch_size=None, sep_h=True,
                       ncolor=None, save_single=False, lg_replace=[],
                       no_title=False):
    run_names = get_run_names_events(logdir, patterns)
    data = get_data_f(logdir, run_names, tag_names, batch_size)
    if len(data) == 0:
        return data, run_names
    num = len(tag_names)
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    if not save_single:
        fig = plt.figure(figsize=(7 * width, 4 * height))
        fig.subplots(height, width)
    else:
        plt.figure(figsize=(7, 4))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    if save_single:
        fig_dir = fig_name[:fig_name.rfind('.')]
        try:
            os.makedirs(fig_dir)
        except os.error:
            pass
    for i in range(len(tag_names)):
        yl = ylim[i]
        if not isinstance(yl, list) and yl is not None:
            yl = ylim
        if not save_single:
            plt.subplot(height, width, fi)
        plot_tag(data, plot_f, list(run_names), tag_names[i], lg_tags, yl,
                 ncolor=ncolor, lg_replace=lg_replace, no_title=no_title)
        if save_single:
            plt.savefig('%s/%s.pdf' % (fig_dir, tag_names[i]),
                        dpi=100, bbox_inches='tight')
            plt.figure(figsize=(7, 4))
        fi += 1
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names
