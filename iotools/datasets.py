from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import torch, h5py
import numpy as np
from torch.utils.data import Dataset

class SparseImage(Dataset):
    
    def __init__(self, data_files, pdg_list=[]):
        """
        Args: data_files ... a list of data files to read
              read_keys ......... a list of string values = data product keys in h5 file to be read-in (besides 'data')
        """
        
        # Validate file paths
        self._files = [f if f.startswith('/') else os.path.join(os.getcwd(),f) for f in data_files]
        for f in self._files:
            if os.path.isfile(f): continue
            sys.stderr.write('File not found:%s\n' % f)
            raise FileNotFoundError

        # Loop over files and scan events
        self._file_handles = [None] * len(self._files)
        self._event_to_file_index  = []
        self._event_to_entry_index = []
        self._shape = None
        search_pdg = not len(pdg_list) 
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r',swmr=True)
            # data size should be common across keys (0th dim)
            data_size = f['data'].shape[0]
            assert not 'semantic' in f or len(f['semantic']) == data_size
            assert not 'pdg' in f or len(f['pdg']) == data_size
            # record or ensure the same data shape
            if self._shape is None:
                self._shape = tuple(f['shape'])
            else:
                assert self._shape == tuple(f['shape']) 
            self._event_to_file_index += [file_index] * data_size
            self._event_to_entry_index += range(data_size)
            # if search_pdg is set and pdg exists in data, append unique pdg list
            if 'pdg' in f and search_pdg:
                pdg_list.append(np.unique(f['pdg']))
            
            f.close()
        # if search_pdg is set then combine an array (per file) of pdg
        if search_pdg and len(pdg_list):
            pdg_list = np.unique(np.concatenate(pdg_list))
            
        self._pdg_list = [int(v) for v in pdg_list]

        
    def __len__(self):
        return len(self._event_to_file_index)

    def __getitem__(self,idx):
        file_index = self._event_to_file_index[idx]
        entry_index = self._event_to_entry_index[idx]
        if self._file_handles[file_index] is None:
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r',swmr=True)
        fh = self._file_handles[file_index]

        data = fh['data'][entry_index]
        mask = np.where(data[:,0]>=0.)
        data = data[mask]
        
        # fill the sparse label (if exists)
        label,pdg = None,None
        if 'semantic' in fh:
            label = fh['semantic'][entry_index][mask]
                
        if 'pdg' in fh:
            pdg = int(fh['pdg'][entry_index])
            if not pdg in self._pdg_list:
                sys.stderr.write('Error: PDG %d not expected (list: %s)\n' % (pdg,self.pdg_list))
                raise ValueError
            label = self._pdg_list.index(pdg)
        
        return dict(data=data,label=label,pdg=pdg,index=idx)
        
class DenseImage2D(SparseImage):
    
    SEMANTIC_BACKGROUND_CLASS=2 # the background class for dense semantic segmentation 
    
    def __init__(self, data_files, reduce_axis=2, pdg_list=[]):
        """
        Args: data_files ... a list of data files to read
        """
        super().__init__(data_files=data_files,pdg_list=pdg_list)

        self._data_buffer  = None
        self._label_buffer = None
        self._reduce_axis  = int(reduce_axis)
        assert self._reduce_axis in [0,1,2]
        
    def __getitem__(self,idx):
        
        res = super().__getitem__(idx)
        data,label,pdg = res['data'],res['label'],res['pdg']
        
        if self._data_buffer is None:
            self._data_buffer  = np.zeros(shape=self._shape,dtype=np.float32)
            self._label_buffer = np.zeros(shape=self._shape,dtype=np.float32)
    
        mask = data[:,:3].astype(np.int32)
        self._data_buffer [:] = 0.
        self._data_buffer[mask[:,0],mask[:,1],mask[:,2]] = data[:,3]
        data  = np.sum(self._data_buffer,axis=self._reduce_axis)
        
        if isinstance(label,np.ndarray):
            self._label_buffer[:] = SEMANTIC_BACKGROUND_CLASS
            self._label_buffer[mask[:,0],mask[:,1],mask[:,2]] = label
            label = np.min(self._data_buffer,axis=self._reduce_axis)
            
        return dict(data=data,label=label,pdg=pdg,index=idx)

