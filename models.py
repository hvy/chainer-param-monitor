import chainer
from chainer import links as L
from chainer import functions as F


class CNN(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            l1=L.Linear(None, 1024),
            l2=L.Linear(None, 10)
        )

    def __call__(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.l1(h)
        h = self.l2(h)
        return h
