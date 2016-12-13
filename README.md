# Parameter Monitoring for Chainer

Statistics for weights, biases, gradiends can be computed during training with this Chainer utility. You can fetch statistics from any chainer model [chainer.Chain](http://docs.chainer.org/en/stable/reference/core/link.html) and repeat this for every iteration or epoch, saving them to a log (e.g. using [chainer.report()](http://docs.chainer.org/en/stable/reference/util/reporter.html)) to plot the statistical changes for a neural network over the course of training.

## Statistics

### Data

- Mean
- Standard deviation
- Min
- Max
- Percentiles
- Sparseness

### Targets

- Weight
- Biases
- Gradients

## Granularity

- Either a specific layer or an aggregation over the entire model.

*Note: It is not yet optimized for speed. Computing percentiles is for instance slow.*

## Example

Weights and biases when training a small convolutional neural network for classification for 100 epochs aggregated over all layers (including final fully connected linear layers). The different alphas show different percentiles.

### Weights

<img src="./samples/weights.png" width="512px;"/>

### Biases

<img src="./samples/biases.png" width="512px;"/>
