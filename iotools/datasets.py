from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import torch, h5py
import numpy as np
from torch.utils.data import Dataset

class SparseImage(Dataset):

    def __init__(self, data_files, pdg_list=[], crop_size=None, angles3d=False):
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
            f.close()

        self._pdg_list = [int(v) for v in pdg_list]

        self._crop = False
        self._csv_handles = [None] * len(self._files)
        if crop_size is not None:
            assert isinstance(crop_size, int) and crop_size > 0 and crop_size <= self._shape[-1]
            self._crop = True
            self._crop_size = crop_size
            self._shape = (crop_size, crop_size, crop_size)
            for file_index, f in enumerate(self._files):
                fcsv = os.path.splitext(f)[0] + ".csv"
                if not os.path.isfile(fcsv):
                    sys.stderr.write('Error: CSV file not found %s' % fcsv)
                    continue

        self._angles3d = angles3d

    def __len__(self):
        return len(self._event_to_file_index)

    def __getitem__(self,idx):
        file_index = self._event_to_file_index[idx]
        entry_index = self._event_to_entry_index[idx]
        if self._file_handles[file_index] is None:
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r',swmr=True)
        if self._csv_handles[file_index] is None:
            fcsv = os.path.splitext(self._files[file_index])[0] + ".csv"
            self._csv_handles[file_index] = np.genfromtxt(fcsv, names=True, delimiter=",")
        fh = self._file_handles[file_index]

        data = fh['data'][entry_index]
        mask = np.where(data[:,0]>=0.)
        data = data[mask]

        if self._crop:
            csv_np = self._csv_handles[file_index][entry_index]
            v0 = np.stack([csv_np['x0'], csv_np['y0'], csv_np['z0']])
            v1 = np.stack([csv_np['x1'], csv_np['y1'], csv_np['z1']])
            diff = - (v0 + v1) / 2. + self._shape[0]/2.
            data[:, :3] = data[:, :3] + diff

            # Compute 3D angles
            v = v1 - v0
            theta = np.arctan2(v[2], np.hypot(v[0], v[1]))
            phi = np.arctan2(v[1], v[0])

        # fill the sparse label (if exists)
        label = None
        if 'semantic' in fh:
            label = fh['semantic'][entry_index][mask]

        if 'pdg' in fh:
            label = int(fh['pdg'][entry_index])
            if not label in self._pdg_list:
                sys.stderr.write('Error: PDG %d not expected (list: %s)\n' % (label,self.pdg_list))
                raise ValueError
            label = self._pdg_list.index(label)

        # Remove pixels that end up outside of crop region
        if self._crop:
            mask = ((data[:, :3] >= 0.) & (data[:, :3] < self._crop_size)).all(axis=1)
            data = data[mask]
            if isinstance(label, np.ndarray):
                label = label[mask]
        if self._angles3d:
            if isinstance(label, np.ndarray):
                pass # TODO?
            else:
                label = [label, theta, phi]

        return (data,label,idx)

class DenseImage2D(SparseImage):

    SEMANTIC_BACKGROUND_CLASS=2 # the background class for dense semantic segmentation

    def __init__(self, data_files, reduce_axis=2, pdg_list=[], crop_size=None, angles3d=False):
        """
        Args: data_files ... a list of data files to read
        """
        super().__init__(data_files=data_files,pdg_list=pdg_list, crop_size=crop_size, angles3d=angles3d)

        self._data_buffer  = None
        self._label_buffer = None
        self._reduce_axis  = int(reduce_axis)
        assert self._reduce_axis in [0,1,2]

    def __getitem__(self,idx):

        data,label,_ = super().__getitem__(idx)

        if self._data_buffer is None:
            self._data_buffer  = np.zeros(shape=self._shape,dtype=np.float32)
            self._label_buffer = np.zeros(shape=self._shape,dtype=np.float32)

        mask = data[:,:3].astype(np.int32)
        self._data_buffer [:] = 0.
        self._data_buffer[mask[:,0],mask[:,1],mask[:,2]] = data[:,3]
        data  = np.sum(self._data_buffer,axis=self._reduce_axis)

        if isinstance(label,np.ndarray):
            self._label_buffer[:] = DenseImage2D.SEMANTIC_BACKGROUND_CLASS
            self._label_buffer[mask[:,0],mask[:,1],mask[:,2]] = label
            label = np.min(self._data_buffer,axis=self._reduce_axis)

        return (data,label,idx)
