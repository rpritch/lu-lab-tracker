
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

from Typing import Any

def save_h5_tracks(path: str, data: Any, meta: dict) -> str:
    """
    Saves napari Points layer data to an HDF5 file in Zephir format.
    Inverts coordinate scaling back to (0, 1) range.
    """
    # 1. Extract data and features from napari
    # Your readers use 'track_id', 'point_id', and 'provenance'
    coords = data  # This is the (N, 4) numpy array: [t, z, y, x]
    features = meta.get('features', {})
    
    worldline_ids = features.get('track_id', np.zeros(len(coords)))
    object_ids = features.get('point_id', np.arange(len(coords)))
    provs = features.get('provenance', ["unknown"] * len(coords))

    # 2. Convert to DataFrame for easier manipulation
    df = pd.DataFrame(coords, columns=['t_idx', 'z', 'y', 'x'])
    
    # 3. Perform Inverse Scaling
    # We look for the volume shape in the layer metadata where the reader stored it
    # Defaulting to 1.0 if not found (no scaling)
    vol_shape = meta.get('metadata', {}).get('vol_shape', [1.0, 1.0, 1.0])
    
    df["z"] /= vol_shape[0]
    df["y"] /= vol_shape[1]
    df["x"] /= vol_shape[2]
    
    # Add metadata columns
    df['worldline_id'] = worldline_ids
    df['id'] = object_ids
    df['provenance'] = [
        p.encode('utf-8') if isinstance(p, str) else p 
        for p in provs
    ]

    # 4. Write to HDF5
    with h5py.File(path, 'w') as f:
        for column in df.columns:
            f.create_dataset(column, data=df[column].values)
        
        # Save the shape attribute so the reader can find it later
        f.attrs["shape"] = vol_shape

    return path


def save_csv_tracks(path: str, data: Any, meta: dict) -> str:
    """
    Saves napari Points layer data to a CSV file.
    Maps internal feature names back to Ascent-style headers.
    """
    # 1. Extract coordinate data (N, 4) -> [t, z, y, x]
    coords = data
    
    # 2. Extract features using the keys defined in your readers
    features = meta.get('features', {})
    
    # Use .get() with fallbacks to prevent errors if features are missing
    track_ids = features.get('track_id', np.zeros(len(coords)))
    object_ids = features.get('point_id', np.arange(len(coords)))
    
    # 3. Create DataFrame with the specific headers from your save_csv() logic
    df = pd.DataFrame(coords, columns=['t', 'z', 'y', 'x'])
    df['TrackID'] = track_ids
    df['ObjectID'] = object_ids
    
    # Optional: Include provenance if it exists in the layer
    if 'provenance' in features:
        df['provenance'] = features['provenance']

    # 4. Save to disk
    # path is provided by the napari save dialog
    df.to_csv(path, index=False)
    
    return path