import numpy as np
from functools import reduce

"""
A collection of functions that extract statistics for given models such as
instances of chainer.Chain in an dictionary.
"""

def get_params(model, param_names=('W', 'b')):
    """Return all parameters (weights and biases) for the given model."""
    params = _flattened_params(model, attr='data')
    report = _as_report(params, prefix=model.name)
    return report


def get_grads(model):
    """Return all currently stored gradient of the model."""
    grads = _flattened_params(model, attr='grad')
    report = _as_report(grads, prefix=model.name + '/grad')
    return report


def get_sparsity(model, param_names=('W', 'b')):
    """Return number of zeros for the given model. """
    def _count_zeros(memo, param):
        if param.name in param_names:
            memo[param.name] += param.data.size - \
                    model.xp.count_nonzero(param.data)
        return memo

    n_zeros = reduce(_count_zeros, model.params(),
                     {pn: 0 for pn in param_names})
    report = {'{}/{}/n_zeros/'.format(model.name, pn): n_zeros[pn] \
              for pn in param_names}

    return report


def _flattened_params(model, param_names=('W', 'b'), attr='data'):
    def _append(memo, param):
        if param.name in param_names:
            # Flatten before appending so that we can concatenate everything
            memo[param.name].append(getattr(param, attr).flatten())
        return memo

    params = reduce(_append, model.params(), {lp: [] for lp in param_names})
    params = {param_name: model.xp.concatenate(param) \
              for (param_name, param) in params.items()}

    return params


def _as_report(data, prefix=''):
    stats = {}
    for name in ['W', 'b']:
        d = data[name]
        stats['{}/{}/min'.format(prefix, name)] = d.min()
        stats['{}/{}/max'.format(prefix, name)] = d.max()
        stats['{}/{}/mean'.format(prefix, name)] = d.mean()
        stats['{}/{}/std'.format(prefix, name)] = d.std()
    return stats
