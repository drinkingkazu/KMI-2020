from . import datasets
from . import collates
from torch.utils.data import DataLoader

def loader_factory(dataset,
                   collate,
                   batch_size,
                   shuffle=True,
                   num_workers=1,
                   **args):
    #ds = getattr(iotools.hkdataset,name)(**args)    
    ds = getattr(datasets,dataset)(**args)
    collate_fn = getattr(collates,collate)
    loader = DataLoader(ds,
                        batch_size  = batch_size,
                        shuffle     = shuffle,
                        num_workers = num_workers,
                        collate_fn  = collate_fn)
    return loader
