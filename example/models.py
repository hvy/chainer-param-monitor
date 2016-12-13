import chainer
from chainer import links as L
from chainer import functions as F

import os, sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import monitor


class CNN(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            l1=L.Linear(None, 1024),
            l2=L.Linear(None, 10)
        )
        self.monitored_layers = ['c1', 'c2', 'c3']

    def __call__(self, x):
        # Collect and report the statistics from the previous call before
        # proceeding with this forward propagation.
        self.report()

        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.l1(h)
        h = self.l2(h)
        return h

    def report(self):
        # To aggregate statistics over all layers, skip the layer argument
        # paramstats = monitor.weight_statistics(self)
        # chainer.report(paramstats)

        for layer in self.monitored_layers:
            stats = monitor.weight_statistics(self, layer)
            chainer.report(stats)

            stats = monitor.bias_statistics(self, layer)
            chainer.report(stats)

            stats = monitor.weight_gradient_statistics(self, layer)
            chainer.report(stats)

            stats = monitor.bias_gradient_statistics(self, layer)
            chainer.report(stats)

