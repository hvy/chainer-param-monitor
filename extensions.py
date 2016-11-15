import os
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chainer.training import extension


def to_tuple(x):
    if x is None:
        return None
    elif isinstance(x, str):
        return x,
    elif not hasattr(x, '__getitem__'):
        return x,
    return x


class ParameterHistogram(extension.Extension):
    def __init__(self, name='main', paramnames=('W', 'b'), layernames=None,
                 dirname='histogram', sample_format='png'):
        self._name = name
        self._paramnames = to_tuple(paramnames)
        self._layernames = to_tuple(layernames)
        self._dirname = dirname
        self._sample_format = sample_format

    def __call__(self, trainer):
        model = trainer.updater.get_optimizer(self._name).target
        xp = model.xp
        data = {pn: xp.array([]) for pn in self._paramnames}

        if self._layernames is not None:  # Only selected layers
            # TODO: Make sure that layers can be specified for arbitrary targets
            for ln in self._layernames:
                for pn in self._paramnames:
                    link = getattr(model, ln)
                    param = getattr(link, pn)
                    # param = getattr(model, '{}/{}'.format(ln, pn))
                    data[pn] = xp.concatenate((data[pn], param.data.flatten()))
        else:  # Aggregate all layers
            for param in model.params():
                pn = param.name
                if pn in self._paramnames:
                    data[pn] = xp.concatenate((data[pn], param.data.flatten()))

        if xp != np:
            for pn, paramdata in data.items():
                data[pn] = xp.asnumpy(paramdata)

        dirname = os.path.join(trainer.out, self._dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = '{}.{}'.format(trainer.updater.epoch, self._sample_format)
        filename = os.path.join(dirname, filename)

        self.save(filename, data)

    def save(self, filename, data, bins=100):
        n_params = len(self._paramnames)
        fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 8), dpi=100)

        for name, ax in zip(self._paramnames, to_tuple(axes)):
            params = data[name]
            title = 'Mean: {0!s:.4s} Std: {1!s:.4s} \nMin: {2!s:.4s} Max: {3!s:.4s}' \
                .format(params.mean(), params.std(), params.min(), params.max())
            weights = np.ones_like(params)/float(len(params))
            ax.hist(params, bins=bins, label=name, alpha=0.5, weights=weights)
            ax.set_title(title)
            ax.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()
