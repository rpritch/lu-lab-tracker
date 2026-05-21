import h5py
import dask.array as da
import os 
import glob
import tifffile
import re
from pathlib import Path
import numpy as np
import pandas as pd

# --- 1. Duck-Typed Class for hdf5 ---
class LazyHDF5Volume:
    def __init__(self, filepath):
        self._file_handle = h5py.File(filepath, 'r')
        self.dask_array = self._build_dask_graph()
        
    def _build_dask_graph(self,ch=None):
        # (Same logic as before to build the nested array)
        def sort_keys(keys):
            return sorted(keys, key=lambda x: int(re.search(r'\d+', x).group()))
        
        t_keys = sort_keys([k for k in self._file_handle.keys() if k.startswith('t')])
        
        time_arrays = []
        for t in t_keys:
            t_group = self._file_handle[t]
            if ch is None:
                c_keys = sort_keys([k for k in t_group.keys() if k.startswith('c')])
            else:
                c_keys = ch
            chan_arrays = [
                da.from_array(t_group[c], chunks=t_group[c].chunks or 'auto') 
                for c in c_keys
            ]
            time_arrays.append(da.stack(chan_arrays, axis=0))
            
        return da.stack(time_arrays, axis=0)

    @property
    def shape(self): return self.dask_array.shape
    @property
    def dtype(self): return self.dask_array.dtype
    @property
    def ndim(self): return self.dask_array.ndim
    def __getitem__(self, key): return self.dask_array[key]

# --- 2. The Worker Function --- #
def read_hdf5(path):
    """
    Reads the filepath and returns a list of napari LayerData tuples.
    """
    print(f"Loading {path} via Custom Dask Reader...")
    
    # Instantiate your lazy volume
    lazy_volume = LazyHDF5Volume(path)
    
    # Define how napari should display this layer
    add_kwargs = {
        "name": "HDF5 Volume",
        "multiscale": False,
        # If your array is (T, C, Z, Y, X), tell napari C is the channel axis
        # so it splits them into separate colored layers automatically!
        "channel_axis": 1 if lazy_volume.ndim == 5 else None, 
    }
    
    # Return exactly one LayerData tuple inside a list
    # Format: (data, meta_dict, layer_type)
    return [(lazy_volume, add_kwargs, 'image')]

def read_sequence(path):
    """
    Reads in a folder of tiff files and lazy loads them
    """
    if not os.path.isdir(path):
        return None
    
    files = sorted(glob.glob(os.path.join(path,"*.tif*")))
    if not files:
        return None
    with tifffile.TiffSequence(files) as seq:
        lazy_volume = da.from_array(seq.asarray(out="memmap"),chunks="auto")
    add_kwargs = {
        "name": "HDF5 Volume",
        "multiscale": False,
        "channel_axis": 1 if lazy_volume.ndim == 5 else None, 
    }
    return [(lazy_volume,add_kwargs,'image')]

def read_h5_points(path):
    """
    Reader for Lu Lab HDF5 annotations.
    Converts normalized Zephir-style coordinates to voxel coordinates.
    """
    path = Path(path)
    
    # 1. Load data from HDF5 into a DataFrame
    with h5py.File(path, "r") as f:
        # Recreating get_annotation_df logic
        data_dict = {k: np.array(f[k]) for k in f.keys()}
        df = pd.DataFrame(data_dict)
        
        # Try to find volume shape in attributes to scale normalized coords
        # Zephir files often store the original volume shape in f.attrs
        vol_shape = f.attrs.get("shape", [1.0, 1.0, 1.0])

    # 2. Extract and transform coordinates (getZephirPointsData logic)
    # We select columns: Time, Z, Y, X
    coords = df[["t_idx", "z", "y", "x"]].to_numpy().astype(float)
    
    # Scale spatial dimensions (z, y, x) by the volume dimensions
    # This converts [0, 1] range to [0, Voxel_Max]
    coords[:, 1] *= vol_shape[0]  # Z
    coords[:, 2] *= vol_shape[1]  # Y
    coords[:, 3] *= vol_shape[2]  # X

    # 3. Prepare Features (Metadata for each point)
    # provenance is often stored as byte strings in HDF5; we decode it for readability
    features = {
        "track_id": df["worldline_id"].to_numpy(),
        "provenance": [
            p.decode('utf-8') if isinstance(p, bytes) else p 
            for p in df["provenance"]
        ],
        "point_id": df["id"].to_numpy(),
    }

    # 4. Define Layer Metadata
    add_kwargs = {
        "name": path.stem,
        "features": features,
        "face_color": "track_id",  # Automatically color points by their track ID
        "face_color_cycle": "viridis",
        "size": 5,
        "ndim": 4,
        "metadata": {"source": path,
                     "vol_shape": vol_shape},
    }

    return [(coords, add_kwargs, "points")]

def read_csv_points(path):
    """
    Reader for CSV annotations. 
    Standardizes various column naming conventions into a unified Points layer.
    """
    path = Path(path)
    
    # 1. Load the CSV (logic from get_annotation_csv)
    df = pd.read_csv(path, header=0)
    
    # Standardize column names
    name_map = {
        "object_id": "id",
        "ObjectID": "id",
        "TrackID": "worldline_id",
        "t": "t_idx"
    }
    df.rename(columns=name_map, inplace=True)
    
    # 2. Ensure essential columns exist
    # If no track/worldline ID exists, treat every point as its own object
    if "worldline_id" not in df.columns:
        df["worldline_id"] = df["id"]
        
    # Add provenance if missing (logic from get_annotation_csv)
    if "provenance" not in df.columns:
        df["provenance"] = "ascent"

    # 3. Extract coordinates and features (logic from getAscentPointsData)
    # Expected order: Time, Z, Y, X
    try:
        coords = df[["t_idx", "z", "y", "x"]].to_numpy().astype(float)
    except KeyError:
        # Fallback for 2D data if 'z' is missing in some CSVs
        coords = df[["t_idx", "y", "x"]].to_numpy().astype(float)

    features = {
        "track_id": df["worldline_id"].to_numpy(),
        "point_id": df["id"].to_numpy(),
        "provenance": df["provenance"].to_numpy(),
    }

    # 4. Define Layer Metadata
    add_kwargs = {
        "name": path.stem,
        "features": features,
        "face_color": "track_id", # Color points by worldline_id
        "face_color_cycle": "viridis",
        "size": 5,
        "metadata": {"source": path},
    }

    return [(coords, add_kwargs, "points")]

# --- 3. The Gatekeeper Functions --- #
def get_h5_reader(path):
    """
    Napari calls this to see if your plugin wants to read the file.
    """
    # If path is a string, check if it ends in .h5 or .hdf5
    if isinstance(path, str) and path.endswith(('.h5', '.hdf5')):
        return read_hdf5
    
    # If path is a list of strings (e.g., user dropped multiple files)
    if isinstance(path, list):
        if all(isinstance(p, str) and p.endswith(('.h5', '.hdf5')) for p in path):
            return read_hdf5

    # Return None to tell napari "I don't know how to read this, try another plugin"
    return None

def get_sequence_reader(path):
    """
    Napari calls this to see if your plugin wants to read the file.
    """
    if not isinstance(path,str) or not os.path.isdir(path):
        return None
    from pathlib import Path
    p = Path(path)
    has_tiffs = any(
        file.suffix.lower() in ('.tif','.tiff')
        for file in p.iterdir() if file.is_file()
    )
    if has_tiffs:
        return read_sequence
    return None

def get_h5_points_reader(path):
    # Standardize everything into a list so we only write the logic once
    paths = [path] if isinstance(path, str) else path
    
    if not isinstance(paths, list):
        return None

    # 1. Check Extensions first (Fast/Cheap)
    valid_exts = ('.h5', '.hdf5')
    if not all(isinstance(p, str) and p.endswith(valid_exts) for p in paths):
        return None

    # 2. Peek inside the files (Slower/Expensive)
    try:
        for p in paths:
            with h5py.File(p, 'r') as f:
                # Check if this H5 is actually an annotation file 
                # by looking for your specific keys
                if "worldline_id" not in f or "t_idx" not in f:
                    return None
        
        # If we made it through the loop without returning None, 
        # all files are valid!
        return read_h5_points
        
    except Exception:
        # If the file is corrupt or not a real HDF5, safely reject it
        return None 

def get_csv_points_reader(path):
    """
    Gatekeeper for CSV annotations.
    Checks for file extension and specific column headers.
    """
    # Standardize input to a list of paths
    paths = [path] if isinstance(path, str) else path
    
    if not isinstance(paths, list):
        return None

    # 1. Fast check: Do all paths end in .csv?
    if not all(isinstance(p, str) and p.endswith('.csv') for p in paths):
        return None

    # 2. Peek inside: Do the files contain our tracking headers?
    try:
        for p in paths:
            # We only read the header (nrows=0) to keep this check lightning fast
            df_peek = pd.read_csv(p, nrows=0)
            
            # Define the 'must-have' columns to distinguish this 
            # from a random CSV file.
            required_columns = {'t_idx', 'x', 'y'}
            
            # If the CSV doesn't have our core coordinates, reject it
            if not required_columns.issubset(df_peek.columns):
                return None
                
        # If all files pass the peek test, return the worker function
        return read_csv_points
        
    except Exception:
        # Catch errors from corrupt files or permission issues
        return None