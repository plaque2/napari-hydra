import os
import numpy as np
from skimage import img_as_float32
from skimage.draw import polygon
import urllib.request
import zipfile
import shutil
import tempfile

import dask.array as da
import napari
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QDoubleSpinBox, QGridLayout, QSizePolicy, QFrame, QApplication, QFileDialog, QMessageBox
from qtpy.QtGui import QPixmap, QIcon
from qtpy.QtCore import Qt

from csbdeep.utils import normalize
from stardist.utils import edt_prob
from stardist.geometry import star_dist
from napari_hydra.hydrastardist.models.model2d_hydra import Config2D, StarDist2D
from napari_hydra.utils import rasterize_labels, ensure_default_model, is_image_stack, process_frame, get_well_centers, sort_wells_grid, calculate_well_stats, calculate_well_diameters, create_hydra_colormaps, get_or_create_layer, prepare_training_batches, write_prediction_summary

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bioimageio_utils")

class HydraStarDistPlugin(QWidget):
    """
    Hydra StarDist Plugin for Napari
    """
    # Default model ZIP to download when no local models are present
    ZIP_URL = "https://rodare.hzdr.de/record/4439/files/model_vacvplaque_hsd.zip?download=1"
    
    # Constants
    WELL_DIAMETER_MM = 35

    def __init__(self, viewer: napari.Viewer = None):
        """
        Initialized the plugin:
            - Loads Hydra StarDist model.
            - Sets up the UI layout.
            - Connects buttons to respective functions.
        """
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Hydra StarDist Plugin")
        plugin_dir = os.path.dirname(__file__)
        
        """
        UI Structure:
            - Header
            - Prediction
                - Selection of image, model, and compression size.
                - Finetuning of thresholds parameters.
            - Counting
                - Counts plaques in each well.
            - Tuning
                - Allows to callibrate the model.
        """
        layout = QVBoxLayout()

        ## HEADER
        ## ---------------------------------------------------------------------------------
        # displays logo and small text
        top_layout = QVBoxLayout()
        top_layout.setAlignment(Qt.AlignHCenter)

        # Logo 
        logo_label = QLabel()
        logo_path = os.path.join(plugin_dir, "resources", "hydrastardist_logo.png")  # makes sure logo exists
        pixmap = QPixmap(logo_path)
        pixmap = pixmap.scaledToWidth(250, mode=Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(logo_label)

        # Small Text
        text_label = QLabel("HydraStarDist is a branched version of StarDist, allowing for star-convex object detection of two different circular targets at the same time.")
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignJustify)
        text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(text_label)

        layout.addLayout(top_layout)
        layout.addSpacing(10)

        ## COMMON ATTRIBUTES
        ## ---------------------------------------------------------------------------------
        # IMAGE AND MODEL
        selection_grid = QGridLayout()
        selection_grid.setHorizontalSpacing(10)
        selection_grid.setVerticalSpacing(5)

        selection_grid.addWidget(QLabel("Image"), 0, 0)
        # Subclass QComboBox to refresh image layers when dropdown is opened
        class RefreshOnShowComboBox(QComboBox):
            def __init__(combo_self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Reference to plugin instance for refresh
                combo_self._plugin = self
            def showPopup(combo_self):
                combo_self._plugin.refresh_image_layers()
                super().showPopup()
        self.image_layer_combo = RefreshOnShowComboBox()
        selection_grid.addWidget(self.image_layer_combo, 1, 0)

        selection_grid.addWidget(QLabel("Model"), 0, 1)
        self.model_combo = QComboBox()
        model_basedir = os.path.join(plugin_dir, "resources", "models")

        # Ensure models directory exists
        os.makedirs(model_basedir, exist_ok=True)

        # If no model subfolders are present, attempt to download the default model zip
        self.model_names = ensure_default_model(model_basedir, HydraStarDistPlugin.ZIP_URL)
        self.model_combo.addItems(self.model_names)

        if "VACV-Plaque" in self.model_names:
            idx = self.model_names.index("VACV-Plaque")
            self.model_combo.setCurrentIndex(idx)
        selection_grid.addWidget(self.model_combo, 1, 1)
        layout.addLayout(selection_grid)
        layout.addSpacing(5)

        if not self.model_names:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Default model download failed.")
            msg.setInformativeText("Please check your internet connection and try again.")
            msg.setWindowTitle("Download Error")
            msg.exec_()
            return

        # LOAD HYDRASTARDIST MODEL
        # searches for model and loads "VACV-Plaque" as default
        self.model = StarDist2D(
            None,
            name="VACV-Plaque",
            basedir=model_basedir
        )

        # connects model combo to change_model slot
        self.model_combo.currentIndexChanged.connect(self.change_model)

        # COMPRESSION SIZE
        target_size_layout = QHBoxLayout()
        target_size_label = QLabel("Resize")
        target_size_layout.addWidget(target_size_label)
        info_icon = QIcon(os.path.join(plugin_dir, "resources", "info.png"))
        info_label = QLabel()
        info_label.setPixmap(info_icon.pixmap(18, 18))
        info_label.setToolTip(
            "Image is compressed for quicker model inference.\n"
            "These are dimensions to which the image will be resized.\n"
            "In case there is a different aspect ratio, the image is\n"
            "cropped and the aspect ratio is kept."
        )
        target_size_layout.addStretch()
        target_size_layout.addWidget(info_label)
        input_size = getattr(self.model.config, "input_size", [800, 600])
        self.target_width_spin = QDoubleSpinBox()
        self.target_width_spin.setRange(1, 4096)
        self.target_width_spin.setDecimals(0)
        self.target_width_spin.setValue(input_size[0])
        self.target_width_spin.setFixedWidth(70)
        self.target_width_spin.setAlignment(Qt.AlignCenter)
        self.target_width_spin.setStyleSheet("padding: 2px;")
        target_size_layout.addWidget(self.target_width_spin)
        target_size_layout.addWidget(QLabel("x"))
        self.target_height_spin = QDoubleSpinBox()
        self.target_height_spin.setRange(1, 4096)
        self.target_height_spin.setDecimals(0)
        self.target_height_spin.setValue(input_size[1])
        self.target_height_spin.setFixedWidth(70)
        self.target_height_spin.setAlignment(Qt.AlignCenter)
        self.target_height_spin.setStyleSheet("padding: 2px;")
        target_size_layout.addWidget(self.target_height_spin)
        layout.addLayout(target_size_layout)
        layout.addSpacing(10)

        ## PREDCITION
        ## ---------------------------------------------------------------------------------
        prediction_header_layout = QHBoxLayout()
        prediction_label = QLabel("<h2>Prediction</h2>")
        prediction_header_layout.addWidget(prediction_label)
        prediction_header_layout.addStretch()
        layout.addLayout(prediction_header_layout)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: gray;")
        layout.addWidget(separator)

        # THRESHOLDING
        threshold_layout = QVBoxLayout()
        reload_icon = QIcon(os.path.join(plugin_dir, "resources", "reload.png"))
        info_icon   = QIcon(os.path.join(plugin_dir, "resources", "info.png"))

        # Table-like grid using QGridLayout
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(5)

        # Threshold Header
        threshold_header_widget = QWidget()
        threshold_header_layout = QHBoxLayout()
        threshold_header_layout.setContentsMargins(0, 0, 0, 0)
        threshold_header_layout.setSpacing(5)
        threshold_label = QLabel("<b>Thresholds</b>")
        threshold_label.setAlignment(Qt.AlignCenter)
        threshold_header_layout.addWidget(threshold_label)

        info_label = QLabel()
        info_label.setPixmap(info_icon.pixmap(18, 18))
        info_label.setToolTip(
            "Thresholds control object detection applied \n"
            "to the HydraStarDist model output."
        )
        threshold_header_layout.addStretch()
        threshold_header_layout.addWidget(info_label)
        threshold_header_widget.setLayout(threshold_header_layout)
        grid.addWidget(threshold_header_widget, 0, 0)

        # Wells Header
        wells_header_widget = QWidget()
        wells_header_layout = QHBoxLayout()
        wells_header_layout.setContentsMargins(0, 0, 0, 0)
        wells_header_layout.setSpacing(5)
        wells_label = QLabel("Wells")
        wells_label.setAlignment(Qt.AlignCenter)
        self.reset_wells_btn = QPushButton()
        self.reset_wells_btn.setIcon(reload_icon)
        self.reset_wells_btn.setToolTip(
            "Reset well thresholds.")
        self.reset_wells_btn.setFixedSize(24, 24)
        self.reset_wells_btn.clicked.connect(self.reset_wells_thresholds)
        wells_header_layout.addWidget(wells_label)
        wells_header_layout.addWidget(self.reset_wells_btn)
        wells_header_widget.setLayout(wells_header_layout)
        grid.addWidget(wells_header_widget, 0, 1)

        # Plaques Header
        plaques_header_widget = QWidget()
        plaques_header_layout = QHBoxLayout()
        plaques_header_layout.setContentsMargins(0, 0, 0, 0)
        plaques_header_layout.setSpacing(5)
        plaques_label = QLabel("Plaques")
        plaques_label.setAlignment(Qt.AlignCenter)
        self.reset_plaques_btn = QPushButton()
        self.reset_plaques_btn.setIcon(reload_icon)
        self.reset_plaques_btn.setToolTip(
            "Reset Plaques thresholds."
        )
        self.reset_plaques_btn.setFixedSize(24, 24)
        self.reset_plaques_btn.clicked.connect(self.reset_plaques_thresholds)
        plaques_header_layout.addWidget(plaques_label)
        plaques_header_layout.addWidget(self.reset_plaques_btn)
        plaques_header_widget.setLayout(plaques_header_layout)
        grid.addWidget(plaques_header_widget, 0, 2)

        # Probability Row
        # Probability Label
        probability_header_widget = QWidget()
        probability_header_layout = QHBoxLayout()
        probability_header_layout.setContentsMargins(0, 0, 0, 0)
        probability_header_layout.setSpacing(5)
        probability_label = QLabel("Probability")
        probability_label.setAlignment(Qt.AlignCenter)
        probability_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        probability_header_layout.addWidget(probability_label)
        
        info_label = QLabel()
        info_label.setPixmap(info_icon.pixmap(18, 18))
        info_label.setToolTip(
            "Defines how strongly the object must resemble a star-convex object."
        )
        probability_header_layout.addStretch()
        probability_header_layout.addWidget(info_label)
        probability_header_widget.setLayout(probability_header_layout)
        grid.addWidget(probability_header_widget, 1, 0)

        #  Well Probability
        self.wells_prob_spin = QDoubleSpinBox()
        self.wells_prob_spin.setRange(0.0, 1.0)
        self.wells_prob_spin.setSingleStep(0.01)
        self.wells_prob_spin.setValue(self.model.thresholds1['prob'])
        self.wells_prob_spin.setAlignment(Qt.AlignCenter)
        self.wells_prob_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.wells_prob_spin.setStyleSheet("padding: 2px;")
        grid.addWidget(self.wells_prob_spin, 1, 1)

        #  Plaque Probability
        self.plaques_prob_spin = QDoubleSpinBox()
        self.plaques_prob_spin.setRange(0.0, 1.0)
        self.plaques_prob_spin.setSingleStep(0.01)
        self.plaques_prob_spin.setValue(self.model.thresholds2['prob'])
        self.plaques_prob_spin.setAlignment(Qt.AlignCenter)
        self.plaques_prob_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plaques_prob_spin.setStyleSheet("padding: 2px;")
        grid.addWidget(self.plaques_prob_spin, 1, 2)

        # NMS Row
        # NMS Label
        nms_header_widget = QWidget()
        nms_header_layout = QHBoxLayout()
        nms_header_layout.setContentsMargins(0, 0, 0, 0)
        nms_header_layout.setSpacing(5)
        nms_label = QLabel("NMS")
        nms_label.setAlignment(Qt.AlignCenter)
        nms_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        nms_header_layout.addWidget(nms_label)

        info_label = QLabel()
        info_label.setPixmap(info_icon.pixmap(18, 18))
        info_label.setToolTip(
            "Defines how strong the non-maximum suppression is."
        )
        nms_header_layout.addStretch()
        nms_header_layout.addWidget(info_label)
        nms_header_widget.setLayout(nms_header_layout)
        grid.addWidget(nms_header_widget, 2, 0)

        # Well NMS
        self.wells_overlap_spin = QDoubleSpinBox()
        self.wells_overlap_spin.setRange(0.0, 1.0)
        self.wells_overlap_spin.setSingleStep(0.01)
        self.wells_overlap_spin.setValue(self.model.thresholds1['nms'])
        self.wells_overlap_spin.setAlignment(Qt.AlignCenter)
        self.wells_overlap_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.wells_overlap_spin.setStyleSheet("padding: 2px;")
        grid.addWidget(self.wells_overlap_spin, 2, 1)

        # Plaque NMS
        self.plaques_overlap_spin = QDoubleSpinBox()
        self.plaques_overlap_spin.setRange(0.0, 1.0)
        self.plaques_overlap_spin.setSingleStep(0.01)
        self.plaques_overlap_spin.setValue(self.model.thresholds2['nms'])
        self.plaques_overlap_spin.setAlignment(Qt.AlignCenter)
        self.plaques_overlap_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plaques_overlap_spin.setStyleSheet("padding: 2px;")
        grid.addWidget(self.plaques_overlap_spin, 2, 2)

        threshold_layout.addLayout(grid)
        layout.addLayout(threshold_layout)
        
        # Prediction Button
        layout.addSpacing(5)
        # Prediction and Export Buttons (side by side)
        prediction_btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.clicked.connect(self.run_prediction)
        prediction_btn_layout.addWidget(self.predict_btn)
        # Add Export button, smaller and next to Run Prediction
        self.export_btn = QPushButton("Export")
        self.export_btn.setFixedWidth(70)
        self.export_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.export_btn.clicked.connect(self.export_prediction)
        prediction_btn_layout.addWidget(self.export_btn)
        layout.addLayout(prediction_btn_layout)

        ## COUNTING
        ## ---------------------------------------------------------------------------------
        layout.addSpacing(10)
        counting_header_layout = QHBoxLayout()
        counting_label = QLabel("<h2>Counting</h2>")
        counting_header_layout.addWidget(counting_label)
        counting_header_layout.addStretch()
        layout.addLayout(counting_header_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: gray;")
        layout.addWidget(separator)
        
        # Grid for Well Counts
        grid_counts = QGridLayout()
        self.well_count_labels = []
        for i in range(6):
            label = QLabel("")
            label.setAlignment(Qt.AlignCenter)
            label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            label.setMinimumWidth(40)
            label.setStyleSheet("background-color: #414851; border-radius: 2px; padding: 2px;")
            self.well_count_labels.append(label)
            grid_counts.addWidget(label, i // 3, i % 3)
        layout.addLayout(grid_counts)

        # Plaque Counts Section
        counts_header_layout = QHBoxLayout()
        counts_label = QLabel("Plaque Count per Well")
        counts_header_layout.addWidget(counts_label)
        counts_header_layout.addStretch()
        self.count_btn = QPushButton("Count")
        self.count_btn.clicked.connect(self.count_plaque)
        counts_header_layout.addWidget(self.count_btn)
        layout.addLayout(counts_header_layout)

        ## TUNING
        ## ---------------------------------------------------------------------------------
        layout.addSpacing(10)
        tuning_header_layout = QHBoxLayout()
        tuning_label = QLabel("<h2>Tuning</h2>")
        tuning_header_layout.addWidget(tuning_label)
        tuning_header_layout.addStretch()
        layout.addLayout(tuning_header_layout)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(2)
        separator.setStyleSheet("background-color: gray;")
        layout.addWidget(separator)

        ## HYPERPARAMETERS
        tuning_grid = QGridLayout()
        tuning_grid.setHorizontalSpacing(10)
        tuning_grid.setVerticalSpacing(5)

        # Batch Size
        tuning_grid.addWidget(QLabel("Batch Size"), 0, 0)
        self.batch_spin = QDoubleSpinBox()
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setSingleStep(1)
        self.batch_spin.setDecimals(0)
        self.batch_spin.setValue(5)
        self.batch_spin.setAlignment(Qt.AlignCenter)
        self.batch_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.batch_spin.setStyleSheet("padding: 2px;")
        tuning_grid.addWidget(self.batch_spin, 1, 0)

        # Epochs
        tuning_grid.addWidget(QLabel("Epochs"), 0, 1)
        self.epochs_spin = QDoubleSpinBox()
        self.epochs_spin.setRange(0, 400)
        self.epochs_spin.setSingleStep(1)
        self.epochs_spin.setDecimals(0)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setAlignment(Qt.AlignCenter)
        self.epochs_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.epochs_spin.setStyleSheet("padding: 2px;")
        tuning_grid.addWidget(self.epochs_spin, 1, 1)

        # Learning Rate
        tuning_grid.addWidget(QLabel("Learning Rate"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0, 1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setAlignment(Qt.AlignCenter)
        self.lr_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lr_spin.setStyleSheet("padding: 2px;")
        tuning_grid.addWidget(self.lr_spin, 1, 2)

        layout.addLayout(tuning_grid)
        layout.addSpacing(5)

        # Tune Button
        self.tune_btn = QPushButton("Tune Model")
        self.tune_btn.clicked.connect(self.tune_model)
        layout.addWidget(self.tune_btn)

        self.setLayout(layout)

        # Fill combo with current image layers
        self.refresh_image_layers()

        # In case labels are uploaded, they are compatible with painting operations
        def ensure_numpy_on_add(event):
            layer = event.value
            if isinstance(layer, Labels) and isinstance(layer.data, da.Array):
                print(f"Converting {layer.name} from dask → numpy (to enable painting)")
                layer.data = layer.data.compute()

        self.viewer.layers.events.inserted.connect(ensure_numpy_on_add)

    def refresh_image_layers(self):
        # Save current selection
        current = self.image_layer_combo.currentText()
        self.image_layer_combo.clear()

        # Only show image layers
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.image_layer_combo.addItems(image_layers)

        # Try to restore previous selection if possible
        if current in image_layers:
            idx = image_layers.index(current)
            self.image_layer_combo.setCurrentIndex(idx)

    def change_model(self):
        selected_model = self.model_combo.currentText()
        plugin_dir = os.path.dirname(__file__)
        model_basedir = os.path.join(plugin_dir, "resources", "models")
        self.model = StarDist2D(
            None,
            name=selected_model,
            basedir=model_basedir
        )
        # Update thresholds spin boxes with new model thresholds
        self.wells_prob_spin.setValue(self.model.thresholds1['prob'])
        self.wells_overlap_spin.setValue(self.model.thresholds1['nms'])
        self.plaques_prob_spin.setValue(self.model.thresholds2['prob'])
        self.plaques_overlap_spin.setValue(self.model.thresholds2['nms'])
        # Update resize spin boxes with model's input size
        input_size = getattr(self.model.config, "input_size", [800, 600])
        self.target_width_spin.setValue(input_size[0])
        self.target_height_spin.setValue(input_size[1])

    def reset_all_thresholds(self):
        """Reset all thresholds to the model's original values."""
        self.wells_prob_spin.setValue(self.model.thresholds1['prob'])
        self.wells_overlap_spin.setValue(self.model.thresholds1['nms'])
        self.plaques_prob_spin.setValue(self.model.thresholds2['prob'])
        self.plaques_overlap_spin.setValue(self.model.thresholds2['nms'])

    def reset_wells_thresholds(self):
        """Reset only Wells thresholds to the model's original values."""
        self.wells_prob_spin.setValue(self.model.thresholds1['prob'])
        self.wells_overlap_spin.setValue(self.model.thresholds1['nms'])

    def reset_plaques_thresholds(self):
        """Reset only Plaques thresholds to the model's original values."""
        self.plaques_prob_spin.setValue(self.model.thresholds2['prob'])
        self.plaques_overlap_spin.setValue(self.model.thresholds2['nms'])

    def _get_well_info(self, wells_data, plaque_data, well_diameter_mm=None):
        """
        Helper to extract well binning, plaque counts, average areas, and well diameters from label arrays.
        Returns:
            counts: list of plaque counts per well (6 bins)
            avg_areas: list of average plaque areas per well (6 bins)
            avg_well_diameter_px: average well diameter in pixels
            px_per_cm: pixel-to-cm scale (using 3.5 cm well diameter)
        """
        if well_diameter_mm is None:
            well_diameter_mm = self.WELL_DIAMETER_MM
        # Convert Dask arrays to numpy if needed
        if isinstance(wells_data, da.Array):
            wells_data = wells_data.compute()
        if isinstance(plaque_data, da.Array):
            plaque_data = plaque_data.compute()
        # Select current frame if stack
        if wells_data.ndim >= 3 and plaque_data.ndim >= 3 and wells_data.shape == plaque_data.shape:
            z_idx = 0
            if hasattr(self.viewer, "dims") and hasattr(self.viewer.dims, "current_step"):
                step = self.viewer.dims.current_step
                if isinstance(step, (tuple, list)) and len(step) > 0:
                    z_idx = step[0]
            z_idx = min(max(z_idx, 0), wells_data.shape[0] - 1)
            wells_2d = wells_data[z_idx]
            plaque_2d = plaque_data[z_idx]
        else:
            wells_2d = wells_data
            plaque_2d = plaque_data
        
        # Get unique well ids and their centers
        well_centers = get_well_centers(wells_2d)
        if not well_centers:
            return [0]*6, [0.0]*6, 0.0, 0.0

        # Sort wells into grid
        ordered_wells = sort_wells_grid(well_centers)

        # Calculate stats
        plaque_counts, avg_areas = calculate_well_stats(wells_2d, plaque_2d, ordered_wells)

        # Compute average well diameter in pixels
        avg_well_diameter_px = calculate_well_diameters(wells_2d, well_centers)

        px_per_cm = avg_well_diameter_px / well_diameter_mm if well_diameter_mm > 0 else 0.0
        return plaque_counts, avg_areas, avg_well_diameter_px, px_per_cm

    def count_plaque(self):
        selected_name = self.image_layer_combo.currentText()
        plaque_layer_name = f"{selected_name} Plaque"
        wells_layer_name = f"{selected_name} Wells"
        if plaque_layer_name not in self.viewer.layers or wells_layer_name not in self.viewer.layers:
            # Clear counts if layers are missing
            for label in self.well_count_labels:
                label.setText("")
            return
        plaque_labels = self.viewer.layers[plaque_layer_name].data
        wells_labels = self.viewer.layers[wells_layer_name].data
        counts, _, _, _ = self._get_well_info(wells_labels, plaque_labels)
        # Update labels
        for i, label in enumerate(self.well_count_labels):
            label.setText(str(counts[i]) if counts else "")

    def _connect_dims_slider(self):
        # Connect the dims slider event to update counts in real-time
        if hasattr(self.viewer, "dims") and hasattr(self.viewer.dims, "events"):
            # Disconnect previous if already connected
            try:
                self.viewer.dims.events.current_step.disconnect(self.count_plaque)
            except Exception:
                pass
            self.viewer.dims.events.current_step.connect(self.count_plaque)



    def predict_single(self, img2d):
        target_width = int(self.target_width_spin.value())
        target_height = int(self.target_height_spin.value())
        h0, w0 = img2d.shape[:2]
        scale = min(w0 / target_width, h0 / target_height)
        new_w = int(w0 / scale)
        new_h = int(h0 / scale)
        y_indices = np.linspace(0, h0 - 1, new_h).astype(int)
        x_indices = np.linspace(0, w0 - 1, new_w).astype(int)
        if img2d.ndim == 3:
            resized = img2d[y_indices[:, None], x_indices[None, :], :]
        else:
            resized = img2d[y_indices[:, None], x_indices[None, :]]
        img = img_as_float32(resized)
        axis_norm = (0, 1)
        img = normalize(img, 1, 99.8, axis=axis_norm)
        labels1, labels2 = self.model.predict_instances(
            img,
            prob_thresh1=self.wells_prob_spin.value(),
            prob_thresh2=self.plaques_prob_spin.value(),
            nms_thresh1=self.wells_overlap_spin.value(),
            nms_thresh2=self.plaques_overlap_spin.value()
        )
        labels1[1]['coord'] = [((coord[0] * scale).astype(int), (coord[1] * scale).astype(int)) for coord in labels1[1]['coord']]
        labels2[1]['coord'] = [((coord[0] * scale).astype(int), (coord[1] * scale).astype(int)) for coord in labels2[1]['coord']]
        labels1[1]['points'] = [((pt[0] * scale).astype(int), (pt[1] * scale).astype(int)) for pt in labels1[1]['points']]
        labels2[1]['points'] = [((pt[0] * scale).astype(int), (pt[1] * scale).astype(int)) for pt in labels2[1]['points']]
        return labels1[1], labels2[1]

    def run_prediction(self):
        selected_name = self.image_layer_combo.currentText()
        if not selected_name:
            print("No image layer selected.")
            return
        # Check if selected layer still exists in the viewer
        if selected_name not in self.viewer.layers:
            print(f"Selected layer '{selected_name}' not found in viewer. Refreshing layer list.")
            self.refresh_image_layers()
            return
        image_layer = self.viewer.layers[selected_name]
        # Convert Dask array to numpy if needed
        if isinstance(image_layer, Labels) and isinstance(image_layer.data, da.Array):
            image_layer.data = image_layer.data.compute()
        image = image_layer.data

        # Detect if input is a stack: [z, y, x] or [z, y, x, c]
        is_stack = False
        if image.ndim == 3 and image.shape[-1] > 3: 
            is_stack = True
        elif image.ndim == 4 and image.shape[0] > 1:
            is_stack = True

        # Get UI resize targets
        target_width = int(self.target_width_spin.value())
        target_height = int(self.target_height_spin.value())
        
        # Prepare colormaps
        white_cmap, blue_cmap = create_hydra_colormaps()

        wells_layer_name = f"{selected_name} Wells"
        plaque_layer_name = f"{selected_name} Plaque"

        # For stacks: add frames incrementally to label layers
        if is_stack:
            n_frames = image.shape[0]
            h, w = image.shape[1], image.shape[2]
            # Create empty arrays to accumulate labels
            wells_labels_stack = np.zeros((n_frames, h, w), dtype=np.int32)
            plaque_labels_stack = np.zeros((n_frames, h, w), dtype=np.int32)
            
            # Add/Get layers
            wells_layer = get_or_create_layer(
                self.viewer, wells_layer_name, wells_labels_stack[:1], white_cmap, "multiplicative"
            )
            plaque_layer = get_or_create_layer(
                self.viewer, plaque_layer_name, plaque_labels_stack[:1], blue_cmap, "additive"
            )

            # Set colormap (in case layer existed)
            wells_layer.colormap = white_cmap
            plaque_layer.colormap = blue_cmap
            plaque_layer.contour = 2
            # For each frame, process and update layer data incrementally
            for z in range(n_frames):
                if image.ndim == 4:
                    this_img = image[z]
                else:
                    this_img = image[z]
                if isinstance(this_img, da.Array):
                    this_img = this_img.compute()
                well_labels, plaque_labels = self.predict_single(this_img)
                if z == 0:
                    self.well_labels = well_labels
                    self.plaque_labels = plaque_labels
                wells_labels_arr, _ = rasterize_labels(well_labels, (h, w))
                plaque_labels_arr, _ = rasterize_labels(plaque_labels, (h, w))
                wells_labels_stack[z] = wells_labels_arr
                plaque_labels_stack[z] = plaque_labels_arr
                # Update layers with current stack up to this frame
                wells_layer.data = wells_labels_stack[:z+1].copy()
                plaque_layer.data = plaque_labels_stack[:z+1].copy()
                # Update the dimension slider to the current frame (z axis)
                if 'z' in self.viewer.dims.axis_labels:
                    z_axis = list(self.viewer.dims.axis_labels).index('z')
                    self.viewer.dims.set_current_step(z_axis, z)
                # Force GUI update
                QApplication.processEvents()
            # After all frames, update layers with full stack and set slider to last
            wells_layer.data = wells_labels_stack
            plaque_layer.data = plaque_labels_stack
            if 'z' in self.viewer.dims.axis_labels:
                    z_axis = list(self.viewer.dims.axis_labels).index('z')
                    self.viewer.dims.set_current_step(z_axis, n_frames - 1)
        else:
            h, w = image.shape[:2]
            well_labels, plaque_labels = self.predict_single(image)
            self.well_labels = well_labels
            self.plaque_labels = plaque_labels
            wells_labels_arr, _ = rasterize_labels(well_labels, (h, w))
            plaque_labels_arr, _ = rasterize_labels(plaque_labels, (h, w))
            
            # Add/Get layers
            wells_layer = get_or_create_layer(
                self.viewer, wells_layer_name, wells_labels_arr, white_cmap, "multiplicative"
            )
            plaque_layer = get_or_create_layer(
                self.viewer, plaque_layer_name, plaque_labels_arr, blue_cmap, "additive"
            )

            wells_layer.colormap = white_cmap
            plaque_layer.colormap = blue_cmap
            plaque_layer.contour = 2
        self.count_plaque()
        # Connect dims slider for real-time updating if stack
        self._connect_dims_slider()
            
    def export_prediction(self):
        selected_name = self.image_layer_combo.currentText()
        if not selected_name:
            QMessageBox.warning(self, "Export Prediction", "No image layer selected.")
            return
        # Check if selected layer still exists in the viewer
        if selected_name not in self.viewer.layers:
            QMessageBox.warning(self, "Export Prediction", f"Selected layer '{selected_name}' not found in viewer.")
            self.refresh_image_layers()
            return

        wells_layer_name = f"{selected_name} Wells"
        plaque_layer_name = f"{selected_name} Plaque"
        if wells_layer_name not in self.viewer.layers or plaque_layer_name not in self.viewer.layers:
            QMessageBox.warning(self, "Export Prediction", "Prediction layers not found.")
            return

        wells_data = self.viewer.layers[wells_layer_name].data
        plaque_data = self.viewer.layers[plaque_layer_name].data

        # Ensure data is numpy
        if isinstance(wells_data, da.Array):
            wells_data = wells_data.compute()
        if isinstance(plaque_data, da.Array):
            plaque_data = plaque_data.compute()

        # Handle stacks
        well_diameter_mm = self.WELL_DIAMETER_MM
        if wells_data.ndim == 3:  # stack: [frames, h, w]
            n_frames = wells_data.shape[0]
            counts_per_frame = []
            avg_areas_per_frame = []
            for z in range(n_frames):
                counts, avg_areas, _, px_per_mm = self._get_well_info(wells_data[z], plaque_data[z], well_diameter_mm=well_diameter_mm)
                counts_per_frame.append(counts)
                avg_areas_mm2 = [area / px_per_mm**2 if px_per_mm > 0 else 0.0 for area in avg_areas]
                avg_areas_per_frame.append(avg_areas_mm2)
        else:  # single image
            counts, avg_areas, _, px_per_mm = self._get_well_info(wells_data, plaque_data, well_diameter_mm=well_diameter_mm)
            counts_per_frame = [counts]
            avg_areas_mm2 = [area / px_per_mm**2 if px_per_mm > 0 else 0.0 for area in avg_areas]
            avg_areas_per_frame = [avg_areas_mm2]

        # Prompt for save location
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prediction Summary", f"{selected_name}_summary.txt", "Text Files (*.txt)"
        )
        if not save_path:
            return

        try:
            write_prediction_summary(save_path, well_diameter_mm, px_per_mm, counts_per_frame, avg_areas_per_frame)
            QMessageBox.information(self, "Export Prediction", f"Prediction summary exported to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Prediction", f"Failed to export summary:\n{e}")

    def tune_model(self):
        """
        Fine-tune the model using the currently selected image and label layers.
        Supports both single images and stacks.
        """
        # LOAD MODEL
        selected_model = self.model_combo.currentText()
        plugin_dir = os.path.dirname(__file__)
        model_basedir = os.path.join(plugin_dir, "resources", "models")
        self.model = StarDist2D(
            None,
            name=selected_model,
            basedir=model_basedir
        )
        config = self.model.config

        # IMAGES & MASKS
        selected_name = self.image_layer_combo.currentText()
        if not selected_name:
            print("No image layer selected.")
            return
        # Check if selected layer still exists in the viewer
        if selected_name not in self.viewer.layers:
            print(f"Selected layer '{selected_name}' not found in viewer. Refreshing layer list.")
            self.refresh_image_layers()
            return
        wells_layer_name = f"{selected_name} Wells"
        plaque_layer_name = f"{selected_name} Plaque"
        if wells_layer_name not in self.viewer.layers or plaque_layer_name not in self.viewer.layers:
            print("Wells or Plaque label layers not found.")
            return
        image_layer = self.viewer.layers[selected_name].data
        wells_layer  = self.viewer.layers[wells_layer_name].data
        plaque_layer = self.viewer.layers[plaque_layer_name].data

        # Convert Dask arrays to numpy
        image = image_layer.compute() if isinstance(image_layer, da.Array) else image_layer
        wells = wells_layer.compute() if isinstance(wells_layer, da.Array) else wells_layer
        plaque = plaque_layer.compute() if isinstance(plaque_layer, da.Array) else plaque_layer

        # Determine if input is a stack
        is_stack = is_image_stack(image)
        
        # If not a stack, wrap single image into a single-frame stack
        if not is_stack:
            # Warn user about single image fine-tuning
            QMessageBox.warning(self, "Single Image Fine-tuning", 
                "Fine-tuning on a single image may result in an overfitted model with poor generalization.")
            # Ensure labels are 2D before wrapping
            img_h, img_w = image.shape[:2]
            img_pixels = img_h * img_w

            
            # Now wrap into single-frame stack
            image = np.expand_dims(image, 0)
            wells = np.expand_dims(wells, 0)
            plaque = np.expand_dims(plaque, 0)
            is_stack = True

        # Get target size from UI
        target_width = int(self.target_width_spin.value())
        target_height = int(self.target_height_spin.value())

        # Prepare batches
        X_train, Y_train = prepare_training_batches(
            image, wells, plaque, is_stack, 
            target_width, target_height, config
        )

        # Get training parameters from UI
        n_frames = image.shape[0]
        batch_size = min(int(self.batch_spin.value()), n_frames) if is_stack else 1
        epochs = int(self.epochs_spin.value())
        learning_rate = float(self.lr_spin.value())

        # Compile and train with specified optimizer and parameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.keras_model.compile(optimizer=optimizer, loss="mse")
        show_info("Tuning the model...")
        self.model.keras_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
        self.model.keras_model.save_weights(os.path.join(model_basedir, selected_model, "weights_best.h5"))
        self.model.keras_model.load_weights(os.path.join(model_basedir, selected_model, "weights_best.h5"))
        show_info("Model tuned!")
        self.run_prediction()
        show_info("Prediction updated!")

def napari_experimental_provide_dock_widget():
    """This function makes the widget discoverable by napari as a plugin dock widget"""
    return HydraStarDistPlugin