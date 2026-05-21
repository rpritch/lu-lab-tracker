import numpy as np
from magicgui import magicgui
from napari.layers import Points, Image
import napari

def make_tracking_widget():
    """
    Creates a widget to manage tracking state and keybindings.
    """

    @magicgui(
        call_button=False,
        current_label={"widget_type": "SpinBox", "value": 1},
    )
    def tracking_controls(viewer: "napari.Viewer", current_label: int):
        pass

    @tracking_controls.parent.changed.connect
    def _init_widget(viewer: "napari.Viewer"):
        # Find the points layer (usually the one loaded by your readers)
        points_layer = None
        for layer in viewer.layers:
            if isinstance(layer, Points):
                points_layer = layer
                break
        
        if points_layer is None:
            return

        # Ensure points are colored by their track_id
        points_layer.face_color = 'track_id'
        points_layer.face_color_cycle = 'viridis'

        # Helper: Get current frame and slice
        def get_dims():
            # dims.current_step is (t, z, y, x)
            return list(viewer.dims.current_step)

        def set_mode_to_select():
            points_layer.mode = 'select'

        # --- KEYMAPPINGS ---

        @points_layer.bind_key('q')
        def dec_frame(viewer):
            step = get_dims()
            step[0] = max(0, step[0] - 1)
            viewer.dims.current_step = step
            set_mode_to_select()

        @points_layer.bind_key('e')
        def inc_frame(viewer):
            step = get_dims()
            step[0] = min(viewer.dims.range[0][1], step[0] + 1)
            viewer.dims.current_step = step
            set_mode_to_select()

        @points_layer.bind_key('s')
        def dec_slice(viewer):
            step = get_dims()
            step[1] = max(0, step[1] - 1)
            viewer.dims.current_step = step

        @points_layer.bind_key('w')
        def inc_slice(viewer):
            step = get_dims()
            step[1] = min(viewer.dims.range[1][1], step[1] + 1)
            viewer.dims.current_step = step

        @points_layer.bind_key('x')
        def dec_label(viewer):
            tracking_controls.current_label.value = max(0, tracking_controls.current_label.value - 1)

        @points_layer.bind_key('c')
        def inc_label(viewer):
            tracking_controls.current_label.value += 1

        @points_layer.bind_key('y')
        def yank_label(viewer):
            if len(points_layer.selected_data) > 0:
                idx = list(points_layer.selected_data)[0]
                val = points_layer.features['track_id'][idx]
                tracking_controls.current_label.value = int(val)

        @points_layer.bind_key('f')
        def find_label(viewer):
            t = get_dims()[0]
            mask = (points_layer.features['track_id'] == tracking_controls.current_label.value) & \
                   (points_layer.data[:, 0] == t)
            if np.any(mask):
                coords = points_layer.data[mask][0]
                viewer.dims.current_step = coords[:2] # Move to T and Z

        @points_layer.bind_key('r')
        def reassign_label(viewer):
            if len(points_layer.selected_data) > 0:
                for idx in points_layer.selected_data:
                    points_layer.features['track_id'][idx] = tracking_controls.current_label.value
                points_layer.refresh_colors()

        @points_layer.bind_key('n')
        def next_label(viewer):
            t_curr = get_dims()[0]
            all_labels = set(points_layer.features['track_id'])
            labels_in_frame = set(points_layer.features['track_id'][points_layer.data[:, 0] == t_curr])
            candidates = sorted(list(all_labels - labels_in_frame))
            if candidates:
                tracking_controls.current_label.value = candidates[0]

        @points_layer.bind_key('m')
        def new_label(viewer):
            max_label = np.max(points_layer.features['track_id']) if len(points_layer.data) > 0 else 0
            tracking_controls.current_label.value = int(max_label + 1)

        @points_layer.bind_key('p')
        def change_provenance(viewer):
            if len(points_layer.selected_data) > 0:
                for idx in points_layer.selected_data:
                    points_layer.features['provenance'][idx] = "manual"
                print("Provenance updated to manual.")

        @points_layer.bind_key('z')
        def move_to_slice(viewer):
            if len(points_layer.selected_data) > 0:
                z_curr = get_dims()[1]
                for idx in points_layer.selected_data:
                    points_layer.data[idx, 1] = z_curr
                points_layer.data = points_layer.data # Trigger visual update

        @points_layer.bind_key('3')
        def mode_select(viewer):
            points_layer.mode = 'select'

        @points_layer.bind_key('4')
        def mode_pan(viewer):
            points_layer.mode = 'pan_zoom'

    return tracking_controls