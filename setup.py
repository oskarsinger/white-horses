from distutils.core import setup

setup(
    name='WhiteHorsesDataFlow',
    version='0.01',
    packages=[
        'whitehorses',
        'whitehorses.errors',
        'whitehorses.loaders',
        'whitehorses.loaders.e4',
        'whitehorses.loaders.at',
        'whitehorses.loaders.readers',
        'whitehorses.loaders.supervised',
        'whitehorses.loaders.simple',
        'whitehorses.loaders.mixture',
        'whitehorses.loaders.rl',
        'whitehorses.servers',
        'whitehorses.servers.rl',
        'whitehorses.servers.batch',
        'whitehorses.servers.minibatch',
        'whitehorses.servers.masks',
        'whitehorses.servers.gram'])
