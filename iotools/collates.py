from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np

def DenseCollate(batch):
    return dict(data  = np.array([sample[0] for sample in batch]),
                label = np.array([sample[1] for sample in batch]),
                index = np.array([sample[2] for sample in batch]) )

def SparseCollate(batch):
    return dict(data  = [sample[0] for sample in batch],
                label = [sample[1] for sample in batch],
                index = [sample[2] for sample in batch]
               )

def SparseCollateSCN(batch):
    result = {}
    concat = np.concatenate
    # data ... (x,y,z,batch,v)
    voxels = concat( [ concat( [sample[0][:,:3],
                                np.full(shape=[len(sample[0]),1],fill_value=batch_id,dtype=np.int32)],
                              axis=1) for batch_id,sample in enumerate(batch)],
                    axis=0)
    
    values = concat([sample[0][:,3:] for sample in batch],axis=0)
    data = concat([voxels,values],axis=1)
    
    # label
    if isinstance(batch[0][1],np.ndarray):
        label = concat( [ concat( [sample[1], 
                                   np.full(shape=[len(sample[1]),1],fill_value=batch_id,dtype=np.int32)],
                                 axis=1) for batch_id, sample in enumerate(batch)],
                       axis=0)
    else:
        label = np.array([sample[1] for sample in batch])

    return dict(data=data,label=label, index=[sample[2] for sample in batch])



