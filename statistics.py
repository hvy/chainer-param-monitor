import numpy as np
import cupy
from functools import reduce


# Statistic dictionaries will consist of items with the following key format.
key_template = '{link_name}/{param_name}/{attr_name}'


def get_statistics(link, param_name, attr_name,
                   statistics=('min', 'max', 'mean', 'std'),
                   percentile_sigmas=(0.13, 2.28, 15.87, 50, 84.13, 97.72,
                                      99.87)):

    key_base = key_template.format(link_name=link.name,
                                   param_name=param_name,
                                   attr_name=attr_name)

    params = flatten_link(link, (param_name,), (attr_name,))
    stats = {}

    if percentile_sigmas:
        percentiles = param_percentiles(params, sigma=percentile_sigmas)
        for i, percentile in enumerate(percentiles):
            stats['{}/percentile/{}'.format(key_base, i)] = percentile

    for s in statistics:
        try:
            stats['{}/{}'.format(key_base, s)] = getattr(params, s)()
        except ValueError:
            # If data is missing from uninitialized model parameters, add
            # NaN placeholders instead of skipping the measurements completely
            # or registering zeros
            stats['{}/{}'.format(key_base, s)] = float('NaN')

    return stats


def get_sparsity(link, include_bias=False):

    param_names = ('W', 'b') if include_bias else ('W',)
    params = flatten_link(link, param_names, ('data',))
    n_zeros = params.size - link.xp.count_nonzero(params)

    key = key_template.format(link_name=link.name,
                              param_name='Wb' if include_bias else 'W',
                              attr_name='zeros')

    return { key: n_zeros }


def flatten_link(link, param_names, attr_names):

    """Flatten link parameters and return a 1-dimensional array located on the
    same device as the link itself.

    Args:
        link (~chainer.Link): Link to flatten.
        param_names (iterable): Parameter names to flatten,
            e.g. ``('W', 'b')``.
        attr_names (iterable): Attributes names to flatten,
            e.g. ``('data', 'grad')``.
    """

    params = []
    for param in link.params():
        if param.name in param_names:
            for attr_name in attr_names:
                p = getattr(param, attr_name)
                p = p.flatten()
                params.append(p)

    return link.xp.concatenate(params)


def param_percentiles(params, sigma):

    """Compute percentiles for given parameters and return an array with the
    same length as the number of elements in ``sigma``.

    Args:
        params (array): 1-dimensional NumPy or CuPy arryay.
        sigma (tuple): Sigmas for which percentiles are computed.

    Returns:
        array: Array of percentiles.
    """

    def _percentiles(_params, _sigma):
        try:
            return np.percentile(_params, _sigma)
        except IndexError:  # Handle uninitialized model parameters
            return np.array((float('NaN'),) * 7)

    # TODO(hvy): Make percentile computation faster for GPUs
    if isinstance(params, cupy.ndarray):
        params = cupy.asnumpy(params)
        return cupy.asarray(_percentiles(params, sigma))

    return _percentiles(params, sigma)
