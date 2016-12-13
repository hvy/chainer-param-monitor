# Parameter Monitoring for Chainer

Statistics for weights, biases, gradiends can be computed during training with this Chainer utility. You can fetch statistics from any chainer model (**chainer.Chainer**) and repeat this for each iteration of epoch and save them to a log (e.g. using [chainer.report()](http://docs.chainer.org/en/stable/reference/util/reporter.html) to plot the statistical changes for a neural network during over the course of training.

## Targets

- Weight
- Biases
- Gradients

## Statistics

- Mean
- Standard deviation
- Min
- Max
- Percentiles
- Sparseness

## Layers

Specify a specific layer by name or aggregate over all layers.

### Sample

Weights and biases when training a small convolutional neural network for classification for 100 epochs aggregated over all layers (including final fully connected linear layers). The different alphas show different percentiles.

#### Weights

<img src="./samples/weights.png" width="512px;"/>

#### Biases

<img src="./samples/biases.png" width="512px;"/>
