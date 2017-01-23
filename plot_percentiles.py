import argparse
import collections
import matplotlib.pyplot as plt
import json


ylabels = ['Weights', 'Biases', 'Weight Gradients', 'Bias Gradients']
log_key_templates = ['predictor/{layer}/W/data/{statistic}',
                     'predictor/{layer}/b/data/{statistic}',
                     'predictor/{layer}/W/grad/{statistic}',
                     'predictor/{layer}/b/grad/{statistic}']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='example/result/log')
    parser.add_argument('--out', type=str, default='plot.png')
    parser.add_argument('--layers', nargs='+', type=str,
                        default=['conv1', 'conv2', 'conv3', 'fc1', 'fc2'])
    return parser.parse_args()


def load_log(filename, keys=None):

    """Parse a JSON file and return a dictionary with the given keys. Each
    key maps to a list of corresponding data measurements in the file."""

    log = collections.defaultdict(list)

    with open(filename) as f:
        for data in json.load(f):  # For each type of data
            if keys is not None:
                for key in keys:
                    log[key].append(data[key])
            else:
                for key, value in data.items():
                    log[key].append(value)
    return log


def plot_percentile_log(filename, log, layer_names, color='green', dpi=100):

    n_rows = len(layer_names)
    n_cols = len(log_key_templates)

    figsize = (1024*n_cols/dpi, 1024*n_rows/dpi)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row in range(n_rows):
        for col in range(n_cols):

            ax = axes[row, col]
            key_template = log_key_templates[col]

            # Min, Max
            pmin_key = key_template.format(layer=layer_names[row],
                    statistic='min')
            pmax_key = key_template.format(layer=layer_names[row],
                    statistic='max')
            pmin = log[pmin_key]
            pmax = log[pmax_key]
            ax.fill_between(range(len(pmin)), pmin, pmax, facecolor=color,
                            alpha=0.2, linewidth=0)

            # Median
            z_key = key_template.format(layer=layer_names[row],
                    statistic='percentile/3')
            z = log[z_key]
            ax.plot(range(len(z)), z, color=color, alpha=0.2)

            # Get all percentiles and fill between
            n_percentiles = 3
            for p in range(n_percentiles):
                s_key = key_template.format(layer=layer_names[row],
                        statistic='percentile/{}'.format(p))
                ns_key = key_template.format(layer=layer_names[row],
                        statistic='percentile/{}'.format(6-p))
                s = log[s_key]
                ns = log[ns_key]
                ax.fill_between(range(len(s)), s, ns, facecolor=color,
                                alpha=0.2, linewidth=0)

            ax.set_xlabel('Epochs')
            ax.set_ylabel(ylabels[col])
            ax.set_title(layer_names[row])

    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.clf()
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    log = load_log(args.log)
    plot_percentile_log(args.out, log, args.layers)
