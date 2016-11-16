import chainer


def params(model, param_names=('W', 'b')):
    xp = model.xp
    data = {pn: xp.array([]) for pn in param_names}

    # Concatenate all desired params from all layers in the given model
    for param in model.params():
        if param.name in param_names:
            data[param.name] = xp.concatenate((data[param.name], param.data.flatten()))

    if xp == chainer.cuda.cupy:
        for pn, params in data.items():
            data[pn] = xp.asnumpy(params)

    observations = {}
    mn = model.name
    for pn, params in data.items():
        observations['{}/{}/min'.format(mn, pn)] = params.min()
        observations['{}/{}/max'.format(mn, pn)] = params.max()
        observations['{}/{}/mean'.format(mn, pn)] = params.mean()
        observations['{}/{}/std'.format(mn, pn)] = params.std()

    return observations
