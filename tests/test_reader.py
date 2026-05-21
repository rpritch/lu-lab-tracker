import os
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
from lu_lab_tracker._reader import (
    get_h5_reader, read_hdf5,
    get_sequence_reader, read_sequence,
    get_h5_points_reader, read_h5_points,
    get_csv_points_reader, read_csv_points
)

# --- FIXTURES FOR DATA GENERATION ---

@pytest.fixture
def h5_image_file(tmp_path):
    """Creates a valid HDF5 image file."""
    path = tmp_path / "image.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 3, 10, 10)))
    return str(path)

@pytest.fixture
def tiff_sequence_dir(tmp_path):
    """Creates a directory with dummy TIFF files."""
    seq_dir = tmp_path / "sequence"
    seq_dir.mkdir()
    for i in range(3):
        tifffile.imwrite(seq_dir / f"frame_{i}.tif", np.zeros((10, 10), dtype=np.uint16))
    return str(seq_dir)

@pytest.fixture
def h5_points_file(tmp_path):
    """Creates a valid HDF5 points/annotation file."""
    path = tmp_path / "points.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("t_idx", data=np.array([0, 1, 2]))
        f.create_dataset("x", data=np.array([0.1, 0.2, 0.3]))
        f.create_dataset("y", data=np.array([0.4, 0.5, 0.6]))
        f.create_dataset("z", data=np.array([0.7, 0.8, 0.9]))
        f.create_dataset("worldline_id", data=np.array([1, 1, 1]))
        f.create_dataset("id", data=np.array([101, 102, 103]))
        f.create_dataset("provenance", data=np.array([b"user1", b"user1", b"user1"]))
        f.attrs["shape"] = [100, 200, 300] # Z, Y, X
    return str(path)

@pytest.fixture
def csv_points_file(tmp_path):
    """Creates a valid CSV points/annotation file with mixed headers."""
    path = tmp_path / "points.csv"
    df = pd.DataFrame({
        "t_idx": [0, 1],
        "x": [10, 20],
        "y": [30, 40],
        "z": [5, 5],
        "TrackID": [1, 1], # To be mapped to worldline_id
        "ObjectID": [100, 101] # To be mapped to id
    })
    df.to_csv(path, index=False)
    return str(path)

# --- TEST SUITE ---

def test_h5_image_reader(h5_image_file):
    # Test Gatekeeper
    reader = get_h5_reader(h5_image_file)
    assert reader == read_hdf5
    
    # Test Gatekeeper with list
    assert get_h5_reader([h5_image_file]) == read_hdf5

    # Test rejection
    assert get_h5_reader("not_a_file.txt") is None

def test_tiff_sequence_reader(tiff_sequence_dir, tmp_path):
    # Test Gatekeeper
    reader = get_sequence_reader(tiff_sequence_dir)
    assert reader == read_sequence

    # Test rejection (not a dir)
    assert get_sequence_reader(str(tmp_path / "missing")) is None
    
    # Test rejection (empty dir)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert get_sequence_reader(str(empty_dir)) is None

def test_h5_points_reader(h5_points_file, h5_image_file):
    # Test Gatekeeper
    assert get_h5_points_reader(h5_points_file) == read_h5_points
    
    # Test rejection (h5 exists but lacks specific point keys)
    assert get_h5_points_reader(h5_image_file) is None

    # Test Worker Data
    layer_data = read_h5_points(h5_points_file)
    assert len(layer_data) == 1
    coords, kwargs, layer_type = layer_data[0]
    
    assert layer_type == "points"
    assert "track_id" in kwargs["features"]
    # Check scaling (z=0.7 * shape[0]=100 = 70.0)
    assert coords[0, 1] == pytest.approx(70.0)
    # Check string decoding
    assert kwargs["features"]["provenance"][0] == "user1"

def test_csv_points_reader(csv_points_file, tmp_path):
    # Test Gatekeeper
    assert get_csv_points_reader(csv_points_file) == read_csv_points
    
    # Test rejection (wrong columns)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col1": [1]}).to_csv(bad_csv)
    assert get_csv_points_reader(str(bad_csv)) is None

    # Test Worker Data
    layer_data = read_csv_points(csv_points_file)
    assert len(layer_data) == 1
    coords, kwargs, layer_type = layer_data[0]
    
    assert layer_type == "points"
    assert "track_id" in kwargs["features"] # Checks name_map logic
    assert np.all(kwargs["features"]["provenance"] == "ascent")