import argparse
from chainer import links as L
from chainer import functions as F
from chainer import Chain, cuda, datasets, iterators, training, optimizers
from chainer.training import extensions

from extensions import LinkMonitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--gpu', type=int, default=2)
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('-B', '--batchsize', type=int, default=128)
    return parser.parse_args()


class CNN(Chain):
    def __init__(self):
        super().__init__(
            conv1=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            conv2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            conv3=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            fc1=L.Linear(None, 1024),
            fc2=L.Linear(1024, 10)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.fc1(h)
        h = self.fc2(h)
        return h


def main(args):
    train, test = datasets.get_mnist(withlabel=True, ndim=3)
    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False,
                                         shuffle=False)

    cnn = CNN()
    links = list(cnn.links(skipself=True))
    model = L.Classifier(cnn)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.epochs, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))  # Default log report
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
                                           'main/accuracy',
                                           'validation/main/loss',
                                           'validation/main/accuracy']))

    trainer.extend(LinkMonitor(links), trigger=(1, 'epoch'))
    # trainer.extend(LinkMonitor(cnn), trigger=(1, 'epoch'))
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    main(args)
