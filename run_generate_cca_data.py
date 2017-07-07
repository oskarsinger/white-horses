import click
import os
import h5py

from whitehorses.loaders.multiview.cca import get_easy_CCAPMLs
from drrobert.file_io import get_timestamped as get_ts

@click.command()
@click.option('--data-dir')
@click.option('--num-data', default=1000)
@click.option('--k', default=1)
@click.option('--ds', default='10 20 30')
@click.option('--lazy', default=True)
def run_things_all_day_bb(
    data_dir,
    num_data,
    k,
    ds,
    lazy):

    loaders = get_easy_CCAPMLs(num_data, k, ds, lazy=lazy) 
    names = ['N', 'k', 'ds']
    vals = [
        str(num_data), 
        str(k),
        '-'.join([str(d) for d in ds])]
    filename = get_ts('easy_cca' + '_' + '_'.join(
        [n + '-' + v for (n, v) in zip(names, values)]) \
        + '.hdf5'
    filepath = os.path.join(data_dir, filename)
    hdf5_repo = h5py.File(filepath, 'w')
    
    for (i, l) in enumerate(loaders):
        hdf5_repo.create_dataset(
            str(i), data=l.get_data())

if __name__=='__main__':
    run_things_all_day_bb()
