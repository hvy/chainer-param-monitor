import argparse
from chainer import links as L
from chainer import optimizers, cuda
from chainer import datasets, iterators, training
from chainer.training import extensions
from models import CNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--gpu', type=int, default=2)
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('-B', '--batchsize', type=int, default=128)
    return parser.parse_args()


def main(args):
    train, test = datasets.get_mnist(withlabel=True, ndim=3)
    train_iter = iterators.SerialIterator(train, args.batchsize)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False,
                                         shuffle=False)

    model = L.Classifier(CNN())

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.epochs, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())  # Default log report
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
                                           'main/accuracy',
                                           'validation/main/loss',
                                           'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    main(args)
