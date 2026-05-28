from __future__ import annotations
import numpy as np
import napari
from napari.layers import Image, Points
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSpinBox, QComboBox, QLabel, QPushButton, QCheckBox
from typing import TYPE_CHECKING

from vispy.color import get_colormap
import warnings

if TYPE_CHECKING:
    import napari.viewer

class TrackingWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        
        # 1. Setup Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 2. UI Elements
        layout.addWidget(QLabel("Target Layer:"))
        self.layer_combo = QComboBox()
        self._update_layer_choices()
        layout.addWidget(self.layer_combo)


        layout.addWidget(QLabel("Current Label:"))
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 9999)
        self.label_spin.setValue(1)

        self.label_spin.valueChanged.connect(self._on_label_changed)
        layout.addWidget(self.label_spin)

        self.backup_checkbox = QCheckBox(text="Backup Changes")
        layout.addWidget(self.backup_checkbox)

        self.save_btn = QPushButton(text="Save Tracks")
        self.save_btn.clicked.connect(self._save_tracks)
        layout.addWidget(self.save_btn)




        # 3. Layer Management Setup
        self.viewer.layers.events.inserted.connect(self._update_layer_choices)
        self.viewer.layers.events.removed.connect(self._update_layer_choices)
        self.layer_combo.currentIndexChanged.connect(self._on_active_layer_change)

        # 3. Ensure a Tracking layer exists
        self._update_layer_choices()
        self._setup_keybindings()

        self._on_active_layer_change()

    def _get_points_layers(self) -> list[Points]:
        """Helper to find all Points layers."""
        return [l for l in self.viewer.layers if isinstance(l, Points)]

    def _update_layer_choices(self, event=None):
        """Syncs the QComboBox with current Points layers."""
        self.layer_combo.clear()
        layers = self._get_points_layers()
        for layer in layers:
            self.layer_combo.addItem(layer.name, layer)

    @property
    def active_layer(self) -> Points | None:
        """Returns the currently selected layer object from the combo box."""
        return self.layer_combo.currentData()
    
    def _on_active_layer_change(self):
        """Configure the layer for tracking when it is selected in the widget."""
        layer = self.active_layer
        if layer is None:
            return
        self.points_layer = layer

        # Ensure 'track_id' exists in features
        # 1. Sync the current spinbox value to the layer's "next point" property
        feature_dict = layer.features
        N = layer.data.shape[0]
        if 'track_id' not in layer.features:
            feature_dict['track_id'] = np.array(np.arange(N),dtype=int)
        if 'object_id' not in layer.features:
            feature_dict['object_id'] = np.array(np.arange(N),dtype=int)
        if 'provenance' not in layer.features:
            feature_dict['provenance'] = np.array(["Loaded"]*N,dtype=str)
        layer.features = feature_dict

        layer.feature_defaults["provenance"] = "GUI"

        layer.events.data.connect(self._on_point_changed)
        self._on_point_changed()
        self._on_label_changed()
        
        # 2. Setup Coloring: Color by track_id using a colormap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            layer.face_color = 'track_id'
            layer.face_color_mode = 'cycle' # Uses a color cycle for distinct IDs
            layer.face_color_cycle = self.get_color_cycle(300)

        # 3. Setup Label Display: Show the track_id on the point
        layer.text = {
            'string': '{track_id}',
            'size': 10,
            'color': 'white',
        }
        @layer.bind_key('r', overwrite=True)
        def reassign_label(_):
            layer = self.active_layer
            if layer and len(layer.selected_data) > 0:
                # Modern napari uses a DataFrame for features
                if 'track_id' not in layer.features:
                    layer.features['track_id'] = np.zeros(len(layer.data), dtype=int)
                
                for idx in layer.selected_data:
                    layer.features.loc[idx, 'track_id'] = self.label_spin.value()
                
                layer.face_color = 'track_id'
                layer.refresh_colors()

        @layer.bind_key('s', overwrite=True)
        def dec_slice(_):
            self._move_dim(1, -1)

    def get_color_cycle(self,size:int):
        return np.concat([np.random.random([size,3]),np.ones([size,1])],axis=1)

    def _on_label_changed(self):
        if self.points_layer is not None:
            self.points_layer.feature_defaults["track_id"] = int(self.label_spin.value())
        return
    
    def _on_point_changed(self):
        if self.points_layer is not None:
            if len(self.points_layer.features["object_id"]) > 0:
                self.points_layer.feature_defaults["object_id"]  = max(self.points_layer.features["object_id"])+1
            else:
                self.points_layer.feature_defaults["object_id"] = 1
        return

    def _setup_keybindings(self):
        """Binds tracking navigation and labeling keys to the viewer."""
        @self.viewer.bind_key('q', overwrite=True)
        def dec_frame(_):
            self._move_dim(0, -1)

        @self.viewer.bind_key('e', overwrite=True)
        def inc_frame(_):
            self._move_dim(0, 1)

        @self.viewer.bind_key('w', overwrite=True)
        def inc_slice(_):
            self._move_dim(1, 1)

        @self.viewer.bind_key('s', overwrite=True)
        def dec_slice(_):
            self._move_dim(1, -1)

        @self.viewer.bind_key('c', overwrite=True)
        def inc_label(_):
            self.label_spin.setValue(self.label_spin.value() + 1)

        @self.viewer.bind_key('x', overwrite=True)
        def dec_label(_):
            self.label_spin.setValue(max(1, self.label_spin.value() - 1))

    def _move_dim(self, dimension: int, delta: int):
        """Helper to navigate Time (0) or Z (1) axes."""
        step = list(self.viewer.dims.current_step)
        if dimension < len(step):
            max_val = int(self.viewer.dims.range[dimension][1])
            step[dimension] = np.clip(step[dimension] + delta, 0, max_val)
            self.viewer.dims.current_step = tuple(step)

    def _save_tracks(self):
        pass