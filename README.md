# Parameter Monitoring for Chainer

Statistics for weights, biases, gradiends can be computed during training with this Chainer plugin. You can collect statistics from any [chainer.Chain](http://docs.chainer.org/en/stable/reference/core/link.html) and repeat this for every iteration or epoch, saving them to a log using e.g. [chainer.report()](http://docs.chainer.org/en/stable/reference/util/reporter.html) to plot the statistical changes over the course of training.

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

### Usage

```python
import monitor

# Assumes you've written Chainer code before
model = MLP()
optimizer.setup(model)
loss = model(x, t)
loss.backward()
optimizer.update()

weight_report = monitor.weight_statistics(model)
chainer.report(weight_report) # Mean, std, min, max, percentiles

bias_report = monitor.bias_statistics(model)
chainer.report(bias_report)

fst_layer_grads = monitor.weight_gradient_statistics(model, layer_name='fc1')
chainer.report(fst_layer_grads)

zeros = monitor.sparsity(model, include_bias=False)
chainer.report(zeros)
```

### Plotting the Statistics

Weights and biases when training a small convolutional neural network for classification for 100 epochs aggregated over all layers (including final fully connected linear layers). The different alphas show different percentiles.

#### Weights

<img src="./samples/weights.png" width="512px;"/>

#### Biases

<img src="./samples/biases.png" width="512px;"/>
