from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_dl_project import load_data as load_dl
from .dataloader_dl_project_seg import load_data as load_dl_seg

def load_data(dataname, batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'dl_seg':
        return load_dl_seg(batch_size, val_batch_size, data_root, num_workers)
    else:
        return load_dl(batch_size, val_batch_size, data_root, num_workers)