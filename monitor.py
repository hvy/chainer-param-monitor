import numpy as np
from functools import reduce
from chainer.cuda import cupy

"""
A collection of functions that extract statistics for given models such as
instances of chainer.Chain in an dictionary.
"""

# The name template of the statistic to collect and include in the report.
# E.g. 'predictor/conv1/W/grad/percentile/sigma_one'
key_template = '{model}/{layer}/{param}/{attr}/{statistic}'


def weight_statistics(model, layer_name=None):
    return parameter_statistics(model, 'W', 'data', layer_name)


def bias_statistics(model, layer_name=None):
    return parameter_statistics(model, 'b', 'data', layer_name)


def weight_gradient_statistics(model, layer_name=None):
    return parameter_statistics(model, 'W', 'grad', layer_name)


def bias_gradient_statistics(model, layer_name=None):
    return parameter_statistics(model, 'b', 'grad', layer_name)


def sparsity(model, include_bias=False, layer_name=None):
    xp = model.xp

    def reduce_count_zeros(acc, param):
        if param.name == 'W' or (include_bias and param.name == 'b'):
            acc += param.data.size - xp.count_nonzero(param.data)
        return acc

    if layer_name is not None:
        sparsity = reduce(reduce_count_zeros, [getattr(model, layer_name)], 0)
    else:
        sparsity = reduce(reduce_count_zeros, model.params(), 0)

    key = key_template.format(model=model.name,
                              layer='*' if layer_name is None else layer_name,
                              param='Wb' if include_bias else 'W' ,
                              attr='sparsity',
                              statistic='zeros')

    return { key: sparsity }


def layer_params(layer, param_name, attr_name):
    params = getattr(layer, param_name)
    params = getattr(params, attr_name)
    return params.flatten()


def layers_params(model, param_name, attr_name):
    xp = model.xp
    params = xp.array([], dtype=xp.float32)

    for param in model.params():
        if param.name == param_name:  # 'W' or 'b'
            values = getattr(param, attr_name)
            values = values.flatten()
            params = xp.concatenate((params, values))  # Slow?

    return params


def parameter_statistics(model, param_name, attr_name, layer_name=None):
    if layer_name is not None:  # Collect statistics for a single layer only
        l = getattr(model, layer_name)
        lp = layer_params(l, param_name, attr_name)
        return as_statistics(lp, model.name, param_name, attr_name,
                             layer_name=layer_name)

    lp = layers_params(model, param_name, attr_name)
    return as_statistics(lp, model.name, param_name, attr_name)


def as_statistics(data, model_name, param_name, attr_name, *, layer_name=None,
                  measures=['min', 'max', 'mean', 'std'],
                  measure_percentiles=True):
    stats = {}

    if layer_name is None:
        layer_name = '*'

    for m in measures:
        key = key_template.format(model=model_name,
                                  layer=layer_name,
                                  param=param_name,
                                  attr=attr_name,
                                  statistic=m)
        stats[key] = getattr(data, m)()

    if measure_percentiles:
        # To CPU before computing the percentiles
        if cupy.get_array_module(data) is cupy:
            data = cupy.asnumpy(data)

        percentiles = np.percentile(data,
                (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87))

        # Back to GPU when percentiles are computed
        if cupy.get_array_module(data) is cupy:
            percentiles = cupy.asarray(percentiles)

        for i, p in enumerate(['n3s', 'n2s', 'n1s', 'z', '1s', '2s', '3s']):
            key = key_template.format(model=model_name,
                                      layer=layer_name,
                                      param=param_name,
                                      attr=attr_name,
                                      statistic='percentile/{}'.format(p))
            stats[key] = percentiles[i]

    return stats
