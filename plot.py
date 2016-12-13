import argparse
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='result/log')
    parser.add_argument('--out', type=str, default='result/log.png')
    #parser.add_argument('--keys', nargs='+', type=str, default=['predictor/W/min', 'predictor/W/max','predictor/W/mean','predictor/W/std'])
    parser.add_argument('--keys', nargs='+', type=str, default=['predictor/W/percentile/n3s', 'predictor/W/percentile/n2s', 'predictor/W/percentile/n1s','predictor/W/percentile/z','predictor/W/percentile/1s', 'predictor/W/percentile/2s', 'predictor/W/percentile/3s', 'predictor/W/min', 'predictor/W/max'])
    #parser.add_argument('--keys', nargs='+', type=str, default=['predictor/b/percentile/n3s', 'predictor/b/percentile/n2s', 'predictor/b/percentile/n1s','predictor/b/percentile/z','predictor/b/percentile/1s', 'predictor/b/percentile/2s', 'predictor/b/percentile/3s', 'predictor/b/min', 'predictor/b/max'])
    return parser.parse_args()


def load_log(filename, keys):
    """Parse a JSON file and return a dictionary with the given keys. Each
    key maps to a list of corresponding data measurements in the file."""
    log = collections.defaultdict(list)
    with open(filename) as f:
        for data in json.load(f):  # For each type of data
            for key in keys:
                log[key].append(data[key])
    return log


def plot_percentile_log(filename, log):
    """Create a plot from the given log and write it to disk as an image."""
    """
    for key, data in log.items():
        plt.plot(range(len(data)), data, label=key)
    """
    for i in range(3):
        sigma = log['predictor/W/percentile/{}s'.format(i + 1)]
        neg_sigma = log['predictor/W/percentile/n{}s'.format(i + 1)]
        plt.fill_between(range(len(sigma)), sigma, neg_sigma, facecolor='green', alpha=0.3, linewidth=0)
    maximum = log['predictor/W/max']
    minimum = log['predictor/W/min']
    plt.fill_between(range(len(maximum)), maximum, minimum, facecolor='green', alpha=0.3, linewidth=0)
    zero_percentile = log['predictor/W/percentile/z']
    plt.plot(range(len(zero_percentile)), zero_percentile, color='green')

    ax = plt.gca()
    # ax.set_ylim([0, 2])
    # ax.legend(loc='best')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_log(filename, log):
    """Create a plot from the given log and write it to disk as an image."""
    for key, data in log.items():
        plt.plot(range(len(data)), data, label=key)

    ax = plt.gca()
    # ax.set_ylim([0, 2])
    # ax.legend(loc='best')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()

def main(args):
    log = load_log(args.log, args.keys)
    plot_percentile_log(args.out, log)


if __name__ == '__main__':
    args = parse_args()
    main(args)
