from . import datasets
from . import collates
from torch.utils.data import DataLoader

def dataset_factory(dataset,**args):
    ds = getattr(datasets,dataset)(**args)
    return ds

def loader_factory(dataset,
                   collate,
                   batch_size,
                   shuffle=True,
                   num_workers=1,
                   **args):
    #ds = getattr(datasets,dataset)(**args)
    ds = dataset_factory(dataset,**args)
    collate_fn = getattr(collates,collate)
    loader = DataLoader(ds,
                        batch_size  = batch_size,
                        shuffle     = shuffle,
                        num_workers = num_workers,
                        collate_fn  = collate_fn)
    return loader
