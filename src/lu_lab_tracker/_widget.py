from __future__ import annotations
import numpy as np
import napari
from napari.layers import Image, Points
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSpinBox, QComboBox, QLabel, QPushButton
from typing import TYPE_CHECKING

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
        self.start_button = QPushButton(text="Start Tracking")
        self.start_button.clicked.connect(self._setup_points_layer)
        layout.addWidget(self.start_button)

        layout.addWidget(QLabel("Current Label:"))
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 9999)
        self.label_spin.setValue(1)
        # --- Sync label changes to the layer --- #
        self.label_spin.valueChanged.connect(self._on_label_changed)
        layout.addWidget(self.label_spin)


        # 3. Layer Management Setup
        self.viewer.layers.events.inserted.connect(self._update_layer_choices)
        self.viewer.layers.events.removed.connect(self._update_layer_choices)
        self.layer_combo.currentIndexChanged.connect(self._on_active_layer_change)

        # 3. Ensure a Tracking layer exists
        self._update_layer_choices()
        self._setup_keybindings()

    def _get_image_layers(self) -> list[Image]:
        """Helper to find all Points layers."""
        return [l for l in self.viewer.layers if isinstance(l, Image)]

    def _update_layer_choices(self, event=None):
        """Syncs the QComboBox with current Points layers."""
        self.layer_combo.clear()
        layers = self._get_image_layers()
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
        if 'track_id' not in layer.features:
            layer.features['track_id'] = np.array([], dtype=int)
        
        # 1. Setup Coloring: Color by track_id using a colormap
        layer.face_color = 'track_id'
        layer.face_color_mode = 'cycle' # Uses a color cycle for distinct IDs
        
        # 2. Setup Label Display: Show the track_id on the point
        layer.text = {
            'string': '{track_id}',
            'size': 12,
            'color': 'white',
            'translation': [0, -10], # Offset label slightly above point
        }
        
        # 3. Sync the current spinbox value to the layer's "next point" property
        self._on_label_changed()

    def _on_label_changed(self):
        if self.points_layer is not None:
            self.points_layer.current_features = {'track_id': self.label_spin.value()}
        return
    
    def _setup_points_layer(self):
        self.points_layer.current_features = {'track_id':self.label_spin.value()}

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

        @self.viewer.bind_key('r', overwrite=True)
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

    def _move_dim(self, dimension: int, delta: int):
        """Helper to navigate Time (0) or Z (1) axes."""
        step = list(self.viewer.dims.current_step)
        if dimension < len(step):
            max_val = int(self.viewer.dims.range[dimension][1])
            step[dimension] = np.clip(step[dimension] + delta, 0, max_val)
            self.viewer.dims.current_step = tuple(step)

