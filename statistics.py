import numpy as np
import cupy
from functools import reduce


key_template = '{link_name}/{param_name}/{attr_name}'


def get_statistics(link, param_name, attr_name,
                   statistics=('min', 'max', 'mean', 'std'),
                   percentile_sigmas=(0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87)):

    key_base = key_template.format(link_name=link.name,
                                   param_name=param_name,
                                   attr_name=attr_name)

    params = get_params(link, param_name, attr_name)

    stats = {}

    if percentile_sigmas:
        percentiles = get_percentiles(params, sigma=percentile_sigmas)
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
    xp = link.xp

    def reduce_count_zeros(acc, param):
        if param.name == 'W' or (include_bias and param.name == 'b'):
            acc += param.data.size - xp.count_nonzero(param.data)
        return acc

    sparsity = reduce(reduce_count_zeros, link.params(), 0)

    key = key_template.format(link_name=link.name,
                              param_name='Wb' if include_bias else 'W',
                              attr_name='zeros')

    return { key: sparsity }


def get_params(link, param_name, attr_name):
    xp = link.xp
    params = xp.array([], dtype=xp.float32)

    # TODO(hvy): Cleaner data collection without using concatenate inside loop
    for param in link.params():
        if param.name == param_name:
            values = getattr(param, attr_name)
            values = values.flatten()
            params = xp.concatenate((params, values))

    return params


def get_percentiles(data, sigma):

    """Compute percentiles for data and return an array with the same length
    as the number of elements in ``sigma``.

    Args:
        data (array): 1-dimensional NumPy or CuPy arryay.
        sigma (tuple): Sigmas for which percentiles are computed.

    Returns:
        array: Array of percentiles.
    """

    def _get_percentiles(_data, _sigma):
        try:
            return np.percentile(_data, _sigma)
        except IndexError:  # Handle uninitialized model parameters
            return np.array((float('NaN'),) * 7)

    if isinstance(data, cupy.ndarray):
        # TODO(hvy): Make percentile computation faster for GPUs
        data = cupy.asnumpy(data)
        return cupy.asarray(_get_percentiles(data, sigma))

    return _get_percentiles(data, sigma)
