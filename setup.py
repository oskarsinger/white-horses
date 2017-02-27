from distutils.core import setup

setup(
    name='whitehorses',
    version='0.01',
    packages=[
        'data',
        'data.errors',
        'data.loaders',
        'data.loaders.e4',
        'data.loaders.at',
        'data.loaders.readers',
        'data.loaders.supervised',
        'data.loaders.simple',
        'data.loaders.mixture',
        'data.loaders.rl',
        'data.servers',
        'data.servers.rl',
        'data.servers.batch',
        'data.servers.minibatch',
        'data.servers.masks',
        'data.servers.gram'])
