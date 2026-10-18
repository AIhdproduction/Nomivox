"""German Qt interface; expensive operations execute in a separate QProcess."""
import json
import os
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,
    QLabel,QLineEdit,QPushButton,QComboBox,QTableWidget,QTableWidgetItem,QCheckBox,QPlainTextEdit,
    QProgressBar,QFileDialog,QMessageBox,QDoubleSpinBox,QSpinBox,QScrollArea,QInputDialog,QFrame,
    QListWidget,QToolButton)
from e57cls.files import atomic_json
from e57cls.gui.eta import RunEstimator, load_profile, save_profile
from e57cls.gui.theme import apply_theme
from e57cls.pipeline.config import PROJECT,DEFAULTS,resolve_config
from e57cls.scene.presets import validate_preset

STRINGS = json.loads((PROJECT/'e57cls/gui/strings_de.json').read_text(encoding='utf-8'))
def tr(key, **values):
    return STRINGS[key].format(**values)


def format_duration(seconds, include_seconds=False):
    minutes, seconds = divmod(max(0,int(seconds)),60)
    hours, minutes = divmod(minutes,60)
    text = f'{hours} h {minutes:02d} min' if hours else f'{minutes} min'
    return text+f' {seconds:02d} s' if include_seconds else text


class MainWindow(QMainWindow):
    def __init__(self, source, auto_start=True):
        super().__init__()
        self.source = Path(source) if source else None
        self.preset = json.loads((PROJECT/'presets/default.json').read_text(encoding='utf-8'))
        self.config = None
        self.inspection_report = None
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.worker_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.preview_process = QProcess(self)
        self.preview_process.setWorkingDirectory(str(PROJECT))
        self.preview_process.finished.connect(self.preview_finished)
        self.preview_process.errorOccurred.connect(self.preview_error)
        self.preview_key = None
        self.preview_process_key = None
        self.pending = b''
        self.started = 0
        self.elapsed_at_event = None
        self.elapsed_event_clock = None
        self.current_fraction = 0
        self.finished_ok = False
        self.input_rejected = False
        self.closing = False
        self.loading = False
        self.overrides = {}
        self.watching = False
        self.phase_starts = {}
        self.phase_key = ''
        self.phase_fraction = 0.
        # A ``scene`` is a sub-step of segmentation.  Keep its display state
        # separate from the root phase used for time estimation, otherwise each
        # scene event resets the stopwatch and makes the estimate far too low.
        self.estimate_phase = ''
        self.estimate_fraction = 0.
        self.estimator = RunEstimator()
        self.timing_profile_path = None
        # Applying the theme here keeps direct MainWindow construction in tests
        # visually identical to the normal launcher path.
        apply_theme()
        self.setWindowTitle(tr('app_title'))
        self.resize(1280,820)
        root = QWidget()
        root.setObjectName('appRoot')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16,16,16,16)
        layout.setSpacing(12)
        header = QFrame()
        header.setObjectName('headerBar')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16,12,16,12)
        wordmark = QLabel('Nomivox')
        wordmark.setObjectName('wordmark')
        subtitle = QLabel(tr('app_subtitle'))
        subtitle.setObjectName('subtitle')
        self.current_file = QLabel(self.source.name if self.source else tr('no_source_file'))
        self.current_file.setObjectName('currentFile')
        header_layout.addWidget(wordmark)
        header_layout.addWidget(subtitle)
        header_layout.addStretch()
        header_layout.addWidget(self.current_file)
        layout.addWidget(header)
        content = QHBoxLayout()
        content.setSpacing(12)
        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(6,8,6,8)
        self.navigation = QListWidget()
        self.navigation.setObjectName('navigation')
        self.navigation.addItems([tr(key) for key in ('tab_input','tab_classes','tab_settings','tab_run')])
        sidebar_layout.addWidget(self.navigation)
        sidebar.setFixedWidth(190)
        content.addWidget(sidebar)
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        content.addWidget(self.tabs,1)
        layout.addLayout(content,1)
        self.input_tab()
        self.classes_tab()
        self.settings_tab()
        self.run_tab()
        self.navigation.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.navigation.setCurrentRow)
        self.navigation.setCurrentRow(0)
        self.bottom = QLabel(tr('ready'))
        self.bottom.setWordWrap(True)
        self.bottom.setToolTip(tr('total_estimate_hint'))
        self.overall = QProgressBar()
        self.overall.setRange(0,1000)
        self.overall.setTextVisible(False)
        layout.addWidget(self.overall)
        layout.addWidget(self.bottom)
        self.setCentralWidget(root)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.set_running(False)
        self.load_table()
        # Opening a file must never immediately consume GPU time.  The first
        # page makes the user choose the intended workflow and classes first.
        self.update_source_ui()
        if self.source:
            QTimer.singleShot(0,self.start_preview)
        if auto_start and self.source:
            self.bottom.setText(tr('setup_required'))

    def tab(self,title):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(12)
        self.tabs.addTab(widget,tr(title))
        return widget,layout

    def button(self,key,callback,layout):
        button = QPushButton(tr(key))
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def launcher_state(self):
        try:
            return json.loads((PROJECT/'.cache/launcher.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            return {}

    def selection_directory(self):
        if self.source:
            return self.source.parent
        return Path(self.launcher_state().get('last_folder',PROJECT))

    def save_last_folder(self):
        if self.source:
            state = self.launcher_state()
            atomic_json(PROJECT/'.cache/launcher.json',{**state,'last_folder':str(self.source.parent)})

    def update_source_ui(self):
        selected = self.source is not None
        self.current_file.setText(self.source.name if selected else tr('no_source_file'))
        self.input_path.setText(str(self.source) if selected else '')
        self.select_button.setText(tr('select_another') if selected else tr('select_source'))
        self.select_button.setProperty('role',None if selected else 'primary')
        self.select_button.style().unpolish(self.select_button)
        self.select_button.style().polish(self.select_button)
        self.no_source_hint.setVisible(not selected)
        if hasattr(self,'bottom'):
            self.bottom.setText(tr('setup_required') if selected else tr('no_source'))
        if hasattr(self,'start_button'):
            self.validate()
        self.update_slam_ui()

    def set_summary_status(self,text,status=None):
        self.summary.setText(text)
        self.summary.setProperty('status',status)
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)

    def clear_inspection(self):
        self.scan_table.setRowCount(0)
        self.metadata.clear()
        if hasattr(self, 'input_mode_status'):
            self.input_mode_status.clear()

    def preview_cache(self):
        return Path(self.cache_path.text()).expanduser().resolve()/'preview'

    def source_preview_key(self):
        if self.source is None:
            return None
        source = self.source.resolve()
        try:
            stat = source.stat()
            value = f'{source}|{stat.st_size}|{stat.st_mtime_ns}'
        except OSError:
            value = f'{source}|missing'
        return sha256(value.encode('utf-8')).hexdigest()

    def preview_directory(self,key):
        return self.preview_cache()/key

    def process_environment(self,cache):
        cache = Path(cache)
        env = QProcessEnvironment.systemEnvironment()
        for key,value in {'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1',
                          'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TEMP':str(cache/'temp'),
                          'TMP':str(cache/'temp')}.items():
            env.insert(key,value)
        (cache/'temp').mkdir(parents=True,exist_ok=True)
        return env

    def preview_report_complete(self,path):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
            return all(key in report for key in ('scan_count','image_count','record_count','scans','images','metadata'))
        except (OSError,ValueError,TypeError):
            return False

    def start_preview(self):
        if self.watching or self.source is None:
            return
        self.stop_preview()
        key = self.source_preview_key()
        if key is None:
            return
        self.preview_key = key
        directory = self.preview_directory(key)
        report_path = directory/'inspect_report.json'
        self.clear_inspection()
        if self.preview_report_complete(report_path):
            if self.read_inspection(report_path):
                return
            shutil.rmtree(directory,ignore_errors=True)
        elif directory.exists():
            shutil.rmtree(directory,ignore_errors=True)
        self.set_summary_status(tr('summary_loading'))
        self.preview_progress.show()
        self.preview_progress.setRange(0,0)
        self.preview_process_key = key
        self.preview_process.setProcessEnvironment(self.process_environment(self.preview_cache().parent))
        self.preview_process.start(sys.executable,['-m','e57cls','inspect',str(self.source),'--out',str(directory),'--sample-points','0'])

    def stop_preview(self):
        self.preview_key = None
        self.preview_process_key = None
        self.preview_progress.hide()
        if self.preview_process.state() != QProcess.ProcessState.NotRunning:
            self.preview_process.kill()
            self.preview_process.waitForFinished(5000)

    def preview_finished(self,code,status):
        key = self.preview_process_key
        self.preview_process_key = None
        if key is None or key != self.preview_key or key != self.source_preview_key():
            return
        self.preview_progress.hide()
        report_path = self.preview_directory(key)/'inspect_report.json'
        if code == 0 and self.preview_report_complete(report_path) and self.read_inspection(report_path):
            return
        details = bytes(self.preview_process.readAllStandardError()).decode(errors='replace').strip()
        if details:
            self.log.appendPlainText(details)
        self.set_summary_status(tr('summary_failed'),'warning')

    def preview_error(self,error):
        if self.preview_process_key is None:
            return
        details = self.preview_process.errorString()
        if details:
            self.log.appendPlainText(details)
        if error == QProcess.ProcessError.FailedToStart:
            self.preview_process_key = None
            self.preview_progress.hide()
            self.set_summary_status(tr('summary_failed'),'warning')

    def card(self, title):
        frame = QFrame()
        frame.setProperty('class','card')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16,14,16,16)
        layout.setSpacing(10)
        heading = QLabel(tr(title))
        heading.setObjectName('cardHeading')
        layout.addWidget(heading)
        return frame,layout

    def configure_table(self, table):
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)

    def settings_row(self, form, label, widget):
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(260)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        form.addRow(label_widget,widget)

    def color_swatch(self, value):
        color = QColor(value)
        if not color.isValid():
            return None
        pixmap = QPixmap(14,14)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(0,0,13,13,3,3)
        painter.end()
        return pixmap

    def set_hints_visible(self, visible):
        self.class_hints.setVisible(visible)
        self.hints_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def input_tab(self):
        _,layout = self.tab('tab_input')
        frame,card = self.card('card_processing_mode')
        mode_form = QFormLayout()
        self.processing_mode = QComboBox()
        self.processing_mode.addItem(tr('mode_classify_and_filter'),'classify_and_filter')
        self.processing_mode.addItem(tr('mode_filter_only'),'filter_only')
        mode_form.addRow(tr('processing_mode'),self.processing_mode)
        self.input_mode = QComboBox()
        self.input_mode.addItem(tr('input_mode_auto'), 'auto')
        self.input_mode.addItem(tr('input_mode_static'), 'static')
        self.input_mode.addItem(tr('input_mode_slam'), 'slam')
        self.input_mode.setCurrentIndex(self.input_mode.findData(DEFAULTS['input_mode']))
        mode_form.addRow(tr('input_mode'), self.input_mode)
        self.mode_hint = QLabel(tr('mode_classify_and_filter_hint'))
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setProperty('class','hint')
        mode_form.addRow('',self.mode_hint)
        card.addLayout(mode_form)
        layout.addWidget(frame)
        frame,card = self.card('card_e57_file')
        path_row = QHBoxLayout()
        self.input_path = QLineEdit(str(self.source) if self.source else '')
        self.input_path.setReadOnly(True)
        self.input_path.setPlaceholderText(tr('no_source_file'))
        path_row.addWidget(self.input_path,1)
        self.select_button = self.button('select_another',self.select_source,path_row)
        card.addLayout(path_row)
        self.no_source_hint = QLabel(tr('no_source_hint'))
        self.no_source_hint.setWordWrap(True)
        self.no_source_hint.setProperty('status','warning')
        card.addWidget(self.no_source_hint)
        self.image_requirement_hint = QLabel(tr('images_required_hint'))
        self.image_requirement_hint.setWordWrap(True)
        self.image_requirement_hint.setProperty('class','hint')
        card.addWidget(self.image_requirement_hint)
        hint = QLabel(tr('automatic_flow'))
        hint.setWordWrap(True)
        hint.setProperty('class','hint')
        card.addWidget(hint)
        layout.addWidget(frame)
        frame,card = self.card('card_overview')
        self.summary = QLabel(tr('summary_no_source'))
        card.addWidget(self.summary)
        self.input_mode_status = QLabel()
        self.input_mode_status.setWordWrap(True)
        self.input_mode_status.setProperty('class','hint')
        card.addWidget(self.input_mode_status)
        self.preview_progress = QProgressBar()
        self.preview_progress.setTextVisible(False)
        self.preview_progress.hide()
        card.addWidget(self.preview_progress)
        self.scan_table = QTableWidget(0,8)
        self.scan_table.setHorizontalHeaderLabels([tr(k) for k in ('scan','name','point_count','images','representation','pose','scene','status')])
        self.scan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scan_table.cellDoubleClicked.connect(self.override_scene)
        self.configure_table(self.scan_table)
        card.addWidget(self.scan_table)
        layout.addWidget(frame,1)
        frame,card = self.card('card_metadata')
        self.metadata = QPlainTextEdit()
        self.metadata.setObjectName('metadataView')
        self.metadata.setReadOnly(True)
        self.metadata.setMaximumHeight(150)
        card.addWidget(self.metadata)
        layout.addWidget(frame)

    def classes_tab(self):
        self.class_widget,layout = self.tab('tab_classes')
        frame,toolbar = self.card('card_class_filter')
        row = QHBoxLayout()
        row.addWidget(QLabel(tr('scene')))
        self.scene_mode = QComboBox()
        for mode in ('both','indoor','outdoor'):
            self.scene_mode.addItem(tr('scene_'+mode),mode)
        row.addWidget(self.scene_mode)
        self.scene_mode.currentIndexChanged.connect(self.filter_classes)
        for key,callback in [('load_preset',self.load_preset),('save_preset',self.save_preset),('duplicate_preset',self.save_preset),('reset_preset',self.reset_preset)]:
            self.button(key,callback,row)
        toolbar.addLayout(row)
        layout.addWidget(frame)
        frame,card = self.card('card_classes')
        self.class_table = QTableWidget(0,9)
        self.class_table.setHorizontalHeaderLabels([tr(k) for k in ('enabled','name','internal_name','code','color','prompts','scene','instance_threshold','mask_threshold')])
        self.class_table.itemChanged.connect(self.edit_class)
        self.configure_table(self.class_table)
        self.class_table.setMinimumHeight(300)
        card.addWidget(self.class_table,1)
        buttons = QHBoxLayout()
        self.button('add_class',self.add_class,buttons)
        self.button('remove_class',self.remove_class,buttons)
        buttons.addStretch()
        card.addLayout(buttons)
        layout.addWidget(frame,3)
        frame,card = self.card('card_reflections')
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        card.addWidget(self.validation)
        self.mirror = QCheckBox(tr('mirror_enable'))
        self.mirror.setChecked(DEFAULTS['mirror_analysis'])
        self.glass = QCheckBox(tr('glass_enable'))
        self.glass.setChecked(DEFAULTS['glass_analysis'])
        self.extra = QCheckBox(tr('extra_no_reflection'))
        card.addWidget(self.mirror)
        card.addWidget(self.glass)
        card.addWidget(self.extra)
        self.hints_toggle = QToolButton()
        self.hints_toggle.setText(tr('toggle_hints'))
        self.hints_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.hints_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.hints_toggle.setCheckable(True)
        card.addWidget(self.hints_toggle)
        self.class_hints = QWidget()
        hints_layout = QVBoxLayout(self.class_hints)
        hints_layout.setContentsMargins(0,0,0,0)
        hints_layout.setSpacing(8)
        hint = QLabel(tr('mirror_limits')+' '+tr('mirror_preservation'))
        hint.setWordWrap(True)
        hint.setProperty('class','hint')
        hints_layout.addWidget(hint)
        prompts_hint = QLabel(tr('prompts_hint'))
        prompts_hint.setWordWrap(True)
        prompts_hint.setProperty('class','hint')
        hints_layout.addWidget(prompts_hint)
        card.addWidget(self.class_hints)
        self.hints_toggle.toggled.connect(self.set_hints_visible)
        self.set_hints_visible(False)
        layout.addWidget(frame)
        self.processing_mode.currentIndexChanged.connect(self.update_processing_mode)
        self.update_processing_mode()

    def settings_tab(self):
        self.settings_widget,layout = self.tab('tab_settings')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        forms = QVBoxLayout(contents)
        forms.setContentsMargins(0,0,0,0)
        forms.setSpacing(12)
        scroll.setWidget(contents)
        layout.addWidget(scroll)
        sections = {}
        for key in ('settings_group_profile_output','settings_group_segmentation','settings_group_tuning_geometry',
                    'settings_group_cleanup','settings_group_scene','settings_group_slam','settings_group_mirror_glass'):
            frame,form = self.card(key)
            forms.addWidget(frame)
            sections[key] = QFormLayout()
            sections[key].setSpacing(8)
            sections[key].setLabelAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
            form.addLayout(sections[key])
        self.profile = QComboBox()
        for key in ('fast','balanced','high'):
            self.profile.addItem(tr('profile_'+key),key)
        self.profile.setCurrentIndex(1)
        self.settings_row(sections['settings_group_profile_output'],tr('profile'),self.profile)
        self.controls = {}
        specifications = {
            'image_scale':(.1,1.,.05),'tile_overlap':(0.,.8,.05),'min_observations':(1,100,1),
            'voxel_size':(.001,1.,.005),'tile_size':(336,1512,16),
            'instance_confidence':(.2,1.,.05),'mask_confidence':(.2,1.,.05),'boundary_margin':(0,16,1),
            'min_winning_share':(0.,1.,.05),'view_size':(64,1512,16),'view_fov':(30,120,5),
            'component_min_voxels':(1,1000,1),'cable_component_min_voxels':(1,1000,1),
            'cable_component_min_span':(.01,10.,.01),'electrical_component_min_voxels':(1,1000,1),
            'electrical_component_min_span':(.01,10.,.01),'class_transfer_radius':(.001,.1,.001),
            'surface_tie_share':(.5,1.,.01),'door_depth_limit':(.05,10.,.05),'door_max_width':(.1,10.,.1),'door_max_height':(.1,10.,.1),
            'geometry_fill_min_area':(.01,1000.,.01),'geometry_fill_max_gap':(.01,2.,.01),'geometry_fill_tolerance':(.001,.2,.001),'geometry_fill_max_planes':(1,2000,1),
            'vegetation_ground_cell_size':(.02,2.,.01),'vegetation_ground_band':(.005,.5,.005),
            'building_boundary_cell_size':(.05,2.,.05),'building_boundary_min_height':(.1,10.,.1),
            'visibility_abs':(.001,.5,.01),'visibility_rel':(0.,.1,.005),'depth_width':(256,8192,256),
            'gpu_memory_fraction':(.1,1.,.05),'scene_sky_outdoor':(0.,1.,.01),
            'scene_ceiling_outdoor_max':(0.,1.,.01),'scene_ceiling_indoor':(0.,1.,.01),
            'scene_sky_indoor_max':(0.,1.,.01),'scene_overhead_range':(1.,30.,1.),
            'scene_plane_tolerance':(.001,.1,.005),'scene_plane_fraction':(.1,1.,.05),
            'scene_ceiling_supported':(0.,1.,.01),'scene_upper_returns_indoor':(0.,1.,.05),
            'scene_open_sky_outdoor':(0.,1.,.01),'scene_open_sky_indoor_max':(0.,1.,.01),
            'scene_sky_confidence':(.2,1.,.05),'scene_ceiling_confidence':(.2,1.,.05),
            'stairs_min_confidence':(.2,1.,.05),'stairs_plane_tolerance':(.001,.2,.001),
            'stairs_planar_fraction':(.1,1.,.05),'stairs_min_rise':(.01,2.,.01),
            'mirror_mask_confidence':(.2,1.,.05),'mirror_fit_tolerance':(.001,.1,.005),
            'mirror_plane_tolerance':(.001,.2,.005),'reflection_prompt_agreement':(1,8,1),
            'mirror_merge_angle':(.1,10.,.5),'mirror_merge_distance':(.001,.1,.005),
            'mirror_image_scale':(.1,1.,.05),'mirror_view_size':(336,1512,16),
            'mirror_max_range':(1.,100.,1.),'mirror_min_pixels':(10,10000,10),'mirror_mask_width':(512,4096,128),
            'mirror_frame_band':(1,60,1),'mirror_frame_band_max':(1,60,1),'reflection_neighbor_radius':(.01,.5,.01),
            'mirror_shape_min_iou':(.1,1.,.01),'mirror_shape_max_growth':(1.,3.,.01),'mirror_edge_search':(0.,.5,.01),
            'slam_max_range_indoor':(1.,100.,1.),'slam_max_range_outdoor':(1.,100.,1.),'slam_max_range':(0.,100.,1.),
            'slam_depth_resolution':(64,8192,64),'slam_splat_max_px':(1,64,1),'slam_depth_tol_abs':(.001,.5,.001),
            'slam_depth_tol_rel':(0.,.1,.001),'slam_station_spacing':(.1,20.,.1),
            'slam_time_window':(.1,600.,.1),'slam_weight_distance_scale':(.1,100.,.1),
            'slam_weight_incidence_power':(0.,8.,.1),'slam_weight_radial_power':(0.,8.,.1)}
        groups = {
            'settings_group_segmentation': {'image_scale','tile_overlap','tile_size','view_size','view_fov','instance_confidence','mask_confidence','boundary_margin','gpu_memory_fraction'},
            'settings_group_tuning_geometry': {'min_observations','voxel_size','min_winning_share','surface_tie_share','door_depth_limit','door_max_width','door_max_height','visibility_abs','visibility_rel','depth_width','class_transfer_radius'},
            'settings_group_cleanup': {'component_min_voxels','cable_component_min_voxels','cable_component_min_span','electrical_component_min_voxels','electrical_component_min_span','vegetation_ground_cell_size','vegetation_ground_band','building_boundary_cell_size','building_boundary_min_height','geometry_fill_min_area','geometry_fill_max_gap','geometry_fill_tolerance','geometry_fill_max_planes','stairs_min_confidence','stairs_plane_tolerance','stairs_planar_fraction','stairs_min_rise'},
            'settings_group_scene': {'scene_sky_outdoor','scene_ceiling_outdoor_max','scene_ceiling_indoor','scene_sky_indoor_max','scene_overhead_range','scene_plane_tolerance','scene_plane_fraction','scene_ceiling_supported','scene_upper_returns_indoor','scene_open_sky_outdoor','scene_open_sky_indoor_max','scene_sky_confidence','scene_ceiling_confidence'},
            'settings_group_slam': {'slam_max_range_indoor','slam_max_range_outdoor','slam_max_range','slam_depth_resolution','slam_splat_max_px','slam_depth_tol_abs','slam_depth_tol_rel','slam_station_spacing','slam_time_window','slam_weight_distance_scale','slam_weight_incidence_power','slam_weight_radial_power'},
            'settings_group_mirror_glass': {'mirror_mask_confidence','mirror_fit_tolerance','mirror_plane_tolerance','reflection_prompt_agreement','mirror_merge_angle','mirror_merge_distance','mirror_image_scale','mirror_view_size','mirror_max_range','mirror_min_pixels','mirror_mask_width','mirror_frame_band','mirror_frame_band_max','reflection_neighbor_radius','mirror_shape_min_iou','mirror_shape_max_growth','mirror_edge_search'},
        }
        for key,(low,high,step) in specifications.items():
            widget = QSpinBox() if isinstance(low,int) else QDoubleSpinBox()
            if isinstance(widget,QDoubleSpinBox):
                widget.setDecimals(3)
            widget.setRange(low,high)
            widget.setSingleStep(step)
            widget.setValue(0 if key == 'slam_max_range' and DEFAULTS[key] is None else DEFAULTS[key])
            self.controls[key] = widget
            group = next(name for name, keys in groups.items() if key in keys)
            self.settings_row(sections[group],tr('setting_'+key),widget)
        self.prompt_batch = QComboBox()
        self.prompt_batch.addItem(tr('prompt_batch_auto'), 'auto')
        for value in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
            self.prompt_batch.addItem(str(value), value)
        self.settings_row(sections['settings_group_segmentation'],tr('setting_prompt_batch'),self.prompt_batch)
        self.profile.currentIndexChanged.connect(self.apply_profile)
        self.output_format = QComboBox()
        for key in ('las','laz'):
            self.output_format.addItem(key.upper()+' 1.4',key)
        self.output_format.addItem(tr('output_format_e57'), 'e57')
        self.output_format.setCurrentIndex(max(0, self.output_format.findData(DEFAULTS['format'])))
        self.output_format.setToolTip(tr('output_format_e57_tooltip'))
        self.settings_row(sections['settings_group_profile_output'],tr('output_format'),self.output_format)
        self.point_reduction = QSpinBox()
        self.point_reduction.setRange(0,95)
        self.point_reduction.setSuffix(' %')
        self.point_reduction.setValue(round(DEFAULTS['point_reduction']*100))
        self.point_reduction.setToolTip(tr('point_reduction_tooltip'))
        self.settings_row(sections['settings_group_profile_output'],tr('point_reduction'),self.point_reduction)
        self.output_density = QComboBox()
        self.output_density.addItem(tr('output_density_reduced'), 'reduced')
        self.output_density.addItem(tr('output_density_original'), 'original')
        self.output_density.setCurrentIndex(max(0, self.output_density.findData(DEFAULTS['output_density'])))
        self.settings_row(sections['settings_group_profile_output'],tr('output_density'),self.output_density)
        self.cache_cleanup = QComboBox()
        for key in ('all', 'keep_masks', 'keep_all'):
            self.cache_cleanup.addItem(tr('cache_cleanup_'+key), key)
        self.cache_cleanup.setCurrentIndex(max(0, self.cache_cleanup.findData(DEFAULTS['cache_cleanup'])))
        self.cache_cleanup.setToolTip(tr('cache_cleanup_reclassify_tooltip'))
        self.settings_row(sections['settings_group_profile_output'],tr('cache_cleanup'),self.cache_cleanup)
        self.point_reduction.valueChanged.connect(self.update_output_density)
        self.update_output_density()
        self.diagnostics = QComboBox()
        for key in ('off','sample','all'):
            self.diagnostics.addItem(tr('diagnostics_'+key),key)
        self.diagnostics.setCurrentIndex(1)
        self.settings_row(sections['settings_group_profile_output'],tr('diagnostics'),self.diagnostics)
        self.cache_path = QLineEdit(str(PROJECT/'.cache'))
        self.gpu = QLineEdit('cuda:0')
        self.settings_row(sections['settings_group_profile_output'],tr('cache_folder'),self.cache_path)
        self.settings_row(sections['settings_group_profile_output'],tr('gpu_device'),self.gpu)
        self.strict = QCheckBox(tr('strict_mode'))
        sections['settings_group_profile_output'].addRow(self.strict)
        self.stairs_priority = QCheckBox(tr('stairs_priority'))
        self.stairs_priority.setChecked(DEFAULTS['stairs_priority'])
        sections['settings_group_cleanup'].addRow(self.stairs_priority)
        self.stairs_geometry_check = QCheckBox(tr('stairs_geometry_check'))
        self.stairs_geometry_check.setChecked(DEFAULTS['stairs_geometry_check'])
        sections['settings_group_cleanup'].addRow(self.stairs_geometry_check)
        self.component_cleanup = QCheckBox(tr('component_cleanup'))
        self.component_cleanup.setChecked(DEFAULTS['component_cleanup'])
        sections['settings_group_cleanup'].addRow(self.component_cleanup)
        self.surface_tie_break = QCheckBox(tr('surface_tie_break'))
        self.surface_tie_break.setChecked(DEFAULTS['surface_tie_break'])
        sections['settings_group_cleanup'].addRow(self.surface_tie_break)
        self.geometry_fill = QCheckBox(tr('geometry_fill'))
        self.geometry_fill.setChecked(DEFAULTS['geometry_fill'])
        sections['settings_group_cleanup'].addRow(self.geometry_fill)
        self.vegetation_base_to_ground = QCheckBox(tr('vegetation_base_to_ground'))
        self.vegetation_base_to_ground.setChecked(DEFAULTS['vegetation_base_to_ground'])
        sections['settings_group_cleanup'].addRow(self.vegetation_base_to_ground)
        self.building_enclosure = QCheckBox(tr('building_enclosure'))
        self.building_enclosure.setChecked(DEFAULTS['building_enclosure'])
        sections['settings_group_cleanup'].addRow(self.building_enclosure)
        self.mirror_shape_regularize = QCheckBox(tr('mirror_shape_regularize'))
        self.mirror_shape_regularize.setChecked(DEFAULTS['mirror_shape_regularize'])
        sections['settings_group_mirror_glass'].addRow(self.mirror_shape_regularize)
        self.slam_time_filter = QCheckBox(tr('slam_time_filter'))
        self.slam_time_filter.setChecked(DEFAULTS['slam_time_filter'])
        sections['settings_group_slam'].addRow(self.slam_time_filter)
        self.slam_diagnostics = QCheckBox(tr('slam_diagnostics'))
        self.slam_diagnostics.setChecked(DEFAULTS['slam_diagnostics'])
        sections['settings_group_slam'].addRow(self.slam_diagnostics)
        forms.addStretch()
        self.input_mode.currentIndexChanged.connect(self.update_slam_ui)
        self.update_slam_ui()

    def run_tab(self):
        _,layout = self.tab('tab_run')
        frame,card = self.card('card_actions')
        buttons = QHBoxLayout()
        self.start_button = self.button('start',self.start_run,buttons)
        self.start_button.setProperty('role','primary')
        self.pause_button = self.button('pause',lambda:self.command('pause'),buttons)
        self.cancel_button = self.button('cancel',lambda:self.command('cancel'),buttons)
        self.cancel_button.setProperty('role','danger')
        self.resume_button = self.button('resume',self.resume,buttons)
        self.saved_button = self.button('resume_saved',self.resume_saved,buttons)
        buttons.addStretch()
        card.addLayout(buttons)
        layout.addWidget(frame)
        frame,card = self.card('card_status')
        self.phase = QLabel(tr('ready'))
        self.phase.setObjectName('cardHeading')
        self.details = QLabel()
        self.phase_progress = QProgressBar()
        self.phase_progress.setRange(0,1000)
        card.addWidget(self.phase)
        card.addWidget(self.details)
        card.addWidget(self.phase_progress)
        layout.addWidget(frame)
        frame,card = self.card('card_log')
        self.log = QPlainTextEdit()
        self.log.setObjectName('logView')
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        card.addWidget(self.log)
        layout.addWidget(frame,1)
        frame,card = self.card('card_results')
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        card.addWidget(self.result_label)
        buttons = QHBoxLayout()
        self.open_output = self.button('open_output',lambda:self.open_file(''),buttons)
        self.open_report = self.button('open_report',lambda:self.open_file('report.html'),buttons)
        self.open_log = self.button('open_log',lambda:self.open_file('log.txt'),buttons)
        buttons.addStretch()
        card.addLayout(buttons)
        layout.addWidget(frame)

    def watch(self,output):
        self.stop_preview()
        self.config = json.loads((Path(output)/'run.json').read_text(encoding='utf-8'))
        self.reset_estimator()
        self.elapsed_at_event = None
        self.elapsed_event_clock = None
        self.watching = True
        self.watch_offset = 0
        self.preset = json.loads(Path(self.config['preset']).read_text(encoding='utf-8'))
        processing_mode = self.config.get('processing_mode','classify_and_filter')
        self.processing_mode.setCurrentIndex(max(0,self.processing_mode.findData(processing_mode)))
        self.input_mode.setCurrentIndex(max(0,self.input_mode.findData(self.config.get('input_mode',DEFAULTS['input_mode']))))
        self.load_table()
        for key,widget in self.controls.items():
            value = self.config.get(key,DEFAULTS[key])
            widget.setValue(0 if key == 'slam_max_range' and value is None else value)
        saved_batch = self.config.get('prompt_batch',DEFAULTS['prompt_batch'])
        self.prompt_batch.setCurrentIndex(max(0,self.prompt_batch.findData(saved_batch)))
        self.mirror.setChecked(self.config.get('mirror_analysis',False))
        self.glass.setChecked(self.config.get('glass_analysis',False))
        self.stairs_priority.setChecked(self.config.get('stairs_priority',False))
        self.stairs_geometry_check.setChecked(self.config.get('stairs_geometry_check',DEFAULTS['stairs_geometry_check']))
        self.component_cleanup.setChecked(self.config.get('component_cleanup',DEFAULTS['component_cleanup']))
        self.surface_tie_break.setChecked(self.config.get('surface_tie_break',DEFAULTS['surface_tie_break']))
        self.geometry_fill.setChecked(self.config.get('geometry_fill',DEFAULTS['geometry_fill']))
        self.vegetation_base_to_ground.setChecked(self.config.get('vegetation_base_to_ground',DEFAULTS['vegetation_base_to_ground']))
        self.building_enclosure.setChecked(self.config.get('building_enclosure',DEFAULTS['building_enclosure']))
        self.mirror_shape_regularize.setChecked(self.config.get('mirror_shape_regularize',DEFAULTS['mirror_shape_regularize']))
        self.slam_time_filter.setChecked(self.config.get('slam_time_filter', DEFAULTS['slam_time_filter']))
        self.slam_diagnostics.setChecked(self.config.get('slam_diagnostics', DEFAULTS['slam_diagnostics']))
        self.output_format.setCurrentIndex(max(0, self.output_format.findData(self.config.get('format',DEFAULTS['format']))))
        self.point_reduction.setValue(round(100*self.config.get('point_reduction',DEFAULTS['point_reduction'])))
        self.output_density.setCurrentIndex(max(0, self.output_density.findData(self.config.get('output_density',DEFAULTS['output_density']))))
        self.cache_cleanup.setCurrentIndex(max(0, self.cache_cleanup.findData(self.config.get('cache_cleanup',DEFAULTS['cache_cleanup']))))
        self.scene_mode.setCurrentIndex(self.scene_mode.findData(self.config['scene_mode']))
        self.read_inspection()
        self.update_slam_ui()
        self.set_running(True)
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.result_label.setText(tr('watch_mode'))
        self.watch_timer = QTimer(self)
        self.watch_timer.timeout.connect(self.poll_watch)
        self.watch_timer.start(1000)
        self.poll_watch()
        self.tabs.setCurrentIndex(3)

    def poll_watch(self):
        path = Path(self.config['output'])/'events.jsonl'
        try:
            with path.open('rb') as stream:
                stream.seek(self.watch_offset)
                while line := stream.readline():
                    if not line.endswith(b'\n'):
                        break
                    self.watch_offset = stream.tell()
                    event = json.loads(line)
                    self.started = time.monotonic()-event.get('elapsed_seconds',0)
                    self.handle_event(event)
            self.pause_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            if self.finished_ok or self.input_rejected:
                self.watching = False
                self.watch_timer.stop()
                self.set_running(False)
            self.update_clock()
        except (OSError,ValueError) as error:
            self.log.appendPlainText(str(error))

    def load_table(self):
        self.loading = True
        self.class_table.setRowCount(len(self.preset['classes']))
        for row,c in enumerate(self.preset['classes']):
            active = QTableWidgetItem()
            active.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
            active.setCheckState(Qt.CheckState.Checked if c.get('enabled',True) else Qt.CheckState.Unchecked)
            self.class_table.setItem(row,0,active)
            values = [c['name_de'],c['id'],str(c['code']),c['color'],'\n'.join(c['prompts']),tr('scene_'+c['scene']),
                      '' if c.get('min_instance_conf') is None else str(c['min_instance_conf']),
                      '' if c.get('min_mask_conf') is None else str(c['min_mask_conf'])]
            for col,value in enumerate(values,1):
                item = QTableWidgetItem(value)
                if col == 4:
                    item.setData(Qt.ItemDataRole.DecorationRole,self.color_swatch(value))
                self.class_table.setItem(row,col,item)
        self.class_table.resizeColumnsToContents()
        self.loading = False
        self.filter_classes()
        self.validate()

    def filter_classes(self):
        mode = self.scene_mode.currentData()
        for row,c in enumerate(self.preset['classes']):
            self.class_table.setRowHidden(row,mode != 'both' and c['scene'] not in (mode,'both'))

    def edit_class(self,item):
        if self.loading:
            return
        c,col = self.preset['classes'][item.row()],item.column()
        if col == 0:
            c['enabled'] = item.checkState() == Qt.CheckState.Checked
        else:
            key = ('','name_de','id','code','color','prompts','scene','min_instance_conf','min_mask_conf')[col]
            value = item.text()
            if key == 'prompts':
                value = value.splitlines()
            elif key == 'scene':
                value = {tr('scene_'+m):m for m in ('both','indoor','outdoor')}.get(value,value)
            elif key == 'code' and value.isdecimal():
                value = int(value)
            elif key.startswith('min_'):
                try:
                    value = float(value.replace(',','.')) if value.strip() else None
                except ValueError:
                    pass
            c[key] = value
            if key == 'color':
                self.loading = True
                item.setData(Qt.ItemDataRole.DecorationRole,self.color_swatch(value))
                self.loading = False
        self.validate()

    def validate(self):
        errors,warnings = validate_preset(self.preset, require_active_classes=self.processing_mode.currentData() == 'classify_and_filter')
        messages = [tr('validation_'+e.split(':')[0])+(': '+e.split(':',1)[1] if ':' in e else '') for e in errors+warnings]
        self.validation.setText('\n'.join(messages) if messages else tr('preset_valid'))
        self.validation.setProperty('status','danger' if errors else 'warning' if warnings else 'success')
        self.validation.style().unpolish(self.validation)
        self.validation.style().polish(self.validation)
        if hasattr(self,'start_button'):
            can_start = self.source is not None and not errors and self.process.state() == QProcess.ProcessState.NotRunning
            self.start_button.setEnabled(can_start)
            self.start_button.setToolTip('' if self.source else tr('start_no_source_tooltip'))
        return not errors

    def add_class(self):
        name,ok = QInputDialog.getText(self,tr('add_class'),tr('class_name'))
        if not ok or not name.strip():
            return
        used = {item['code'] for item in self.preset['classes']}
        suggested = next((code for code in range(23,61) if code not in used),23)
        code,ok = QInputDialog.getInt(self,tr('add_class'),tr('class_code'),suggested,23,60)
        if not ok:
            return
        if code in used:
            QMessageBox.warning(self,tr('app_title'),tr('class_code_taken',code=code))
            return
        normalized = ''.join(char if char.isalnum() else '_' for char in name.lower()).strip('_')
        ident = (normalized or 'class')+'_'+uuid4().hex[:6]
        self.preset['classes'].append(dict(id=ident,name_de=name.strip(),name_en=name.strip(),code=code,
            color='#888888',scene='both',enabled=True,prompts=['object'],min_instance_conf=None,min_mask_conf=None))
        self.load_table()

    def update_processing_mode(self):
        filter_only = self.processing_mode.currentData() == 'filter_only'
        self.mode_hint.setText(tr('mode_filter_only_hint') if filter_only else tr('mode_classify_and_filter_hint'))
        if hasattr(self,'class_widget'):
            self.class_widget.setEnabled(not filter_only and self.process.state() == QProcess.ProcessState.NotRunning)
            self.mirror.setChecked(True if filter_only else self.mirror.isChecked())
            self.mirror.setEnabled(not filter_only)
            self.glass.setChecked(True if filter_only else self.glass.isChecked())
            self.glass.setEnabled(not filter_only)
            self.extra.setEnabled(not filter_only)
        if hasattr(self,'start_button'):
            self.validate()

    def update_slam_ui(self):
        """Disable origin-dependent reflection controls for selected SLAM input."""
        if not hasattr(self, 'mirror'):
            return
        detected = self.inspection_report and self.inspection_report.get('input_mode', {}).get('mode') == 'slam'
        slam = self.input_mode.currentData() == 'slam' or (self.input_mode.currentData() == 'auto' and detected)
        settings = [widget for key, widget in getattr(self, 'controls', {}).items()
                    if key.startswith('mirror_') or key == 'reflection_prompt_agreement']
        for widget in (self.mirror, self.glass, self.extra, self.mirror_shape_regularize, *settings):
            widget.setEnabled(not slam and self.processing_mode.currentData() != 'filter_only')
        self.validation.setText(tr('slam_reflection_hint') if slam else '')

    def remove_class(self):
        row = self.class_table.currentRow()
        if row >= 0:
            self.preset['classes'].pop(row)
            self.load_table()

    def load_preset(self):
        path,_ = QFileDialog.getOpenFileName(self,tr('load_preset'),str(PROJECT/'presets'),'JSON (*.json)')
        if path:
            try:
                value = json.loads(Path(path).read_text(encoding='utf-8'))
                errors,_ = validate_preset(value)
                if errors:
                    raise ValueError('; '.join(errors))
                self.preset = value
                self.load_table()
            except Exception as error:
                self.log.appendPlainText(str(error))
                QMessageBox.warning(self,tr('app_title'),tr('preset_invalid'))

    def save_preset(self):
        if not self.validate():
            return
        name,ok = QInputDialog.getText(self,tr('save_preset'),tr('preset_name'))
        if ok and name.strip():
            self.preset['preset_name'] = name.strip()
            path = Path(self.cache_path.text())/'presets'/('preset_'+uuid4().hex[:10]+'.json')
            atomic_json(path,self.preset)
            self.result_label.setText(tr('preset_saved',path=path))

    def reset_preset(self):
        self.preset = json.loads((PROJECT/'presets/default.json').read_text(encoding='utf-8'))
        self.load_table()

    def apply_profile(self):
        scale,overlap,observations = {'fast':(.6,.2,1),'balanced':(.9,.3,1),'high':(1.,.35,1)}[self.profile.currentData()]
        for key,value in [('image_scale',scale),('tile_overlap',overlap),('min_observations',observations)]:
            self.controls[key].setValue(value)

    def update_output_density(self):
        identical = self.point_reduction.value() == 0
        self.output_density.setEnabled(not identical)
        if identical:
            self.output_density.setCurrentIndex(self.output_density.findData('reduced'))

    def select_source(self):
        path,_ = QFileDialog.getOpenFileName(self,tr('select_input'),str(self.selection_directory()),'E57 (*.e57)')
        if path:
            self.source = Path(path).resolve()
            self.config = None
            self.overrides = {}
            self.inspection_report = None
            self.save_last_folder()
            self.update_source_ui()
            self.start_preview()

    def override_scene(self,row,column):
        if column != 6 or self.process.state() != QProcess.ProcessState.NotRunning:
            return
        modes = ('both','indoor','outdoor')
        names = [tr('scene_'+k) for k in modes]
        mode,ok = QInputDialog.getItem(self,tr('scene_override'),tr('scene'),names,0,False)
        if ok:
            self.overrides[str(row+1)] = modes[names.index(mode)]
            self.scan_table.setItem(row,6,QTableWidgetItem(mode))

    def start_run(self):
        if self.source is None or self.watching or self.process.state() != QProcess.ProcessState.NotRunning or not self.validate():
            return
        try:
            self.stop_preview()
            settings = {key:widget.value() for key,widget in self.controls.items()}
            settings['slam_max_range'] = settings['slam_max_range'] or None
            settings['prompt_batch'] = self.prompt_batch.currentData()
            settings.update(source=str(self.source),processing_mode=self.processing_mode.currentData(),input_mode=self.input_mode.currentData(),scene_mode=self.scene_mode.currentData(),profile=self.profile.currentData(),
                mirror_analysis=self.mirror.isChecked(),glass_analysis=self.glass.isChecked(),
                stairs_priority=self.stairs_priority.isChecked(),stairs_geometry_check=self.stairs_geometry_check.isChecked(),component_cleanup=self.component_cleanup.isChecked(),
                surface_tie_break=self.surface_tie_break.isChecked(),geometry_fill=self.geometry_fill.isChecked(),
                vegetation_base_to_ground=self.vegetation_base_to_ground.isChecked(),
                building_enclosure=self.building_enclosure.isChecked(),
                mirror_shape_regularize=self.mirror_shape_regularize.isChecked(),
                slam_time_filter=self.slam_time_filter.isChecked(),slam_diagnostics=self.slam_diagnostics.isChecked(),
                extra_no_reflection=self.extra.isChecked(),format=self.output_format.currentData(),
                point_reduction=self.point_reduction.value()/100,output_density=self.output_density.currentData(),
                diagnostics=self.diagnostics.currentData(),cache_cleanup=self.cache_cleanup.currentData(),cache=self.cache_path.text(),gpu_device=self.gpu.text(),strict=self.strict.isChecked(),
                scene_overrides=self.overrides)
            self.config = resolve_config(settings)
            output = Path(self.config['output'])
            if output.exists():
                output = output.with_name(output.name+'_'+uuid4().hex[:6])
                self.config['output'] = str(output)
            output.mkdir(parents=True,exist_ok=False)
            atomic_json(output/'preset.json',self.preset)
            self.config['preset'] = str(output/'preset.json')
            atomic_json(output/'run.json',self.config)
            state = self.launcher_state()
            atomic_json(PROJECT/'.cache/launcher.json',{**state,'last_folder':str(self.source.parent),'last_config':str(output/'run.json')})
            self.launch_worker()
        except Exception as error:
            self.log.appendPlainText(str(error))
            self.phase.setText(tr('launch_failed'))

    def launch_worker(self):
        self.cache_cleanup_freed = 0
        self.pending = b''
        self.started = time.monotonic()
        self.elapsed_at_event = None
        self.elapsed_event_clock = None
        self.current_fraction = 0
        self.phase_starts = {}
        self.phase_fraction = 0.
        self.phase_key = ''
        self.estimate_phase = ''
        self.estimate_fraction = 0.
        self.reset_estimator()
        self.finished_ok = False
        self.input_rejected = False
        self.process.setProcessEnvironment(self.process_environment(self.config['cache']))
        self.set_running(True)
        self.phase.setText(tr('state_running'))
        self.result_label.setText(tr('result_location',path=self.config['output']))
        self.tabs.setCurrentIndex(3)
        self.process.start(sys.executable,['-m','e57cls','run','--config',str(Path(self.config['output'])/'run.json')])

    def reset_estimator(self):
        cache = Path(self.config['cache'])
        self.timing_profile_path = cache/'timing_profile.json'
        parallel_shared = (self.config.get('processing_mode','classify_and_filter') == 'classify_and_filter'
                           and bool(self.config.get('mirror_analysis',False))
                           and bool(self.config.get('pipeline_parallel',DEFAULTS['pipeline_parallel'])))
        applies = {
            # In the parallel pipeline SAM and local mirror fitting overlap the
            # scan-paced CPU work.  Counting them separately would double-count
            # the ETA, so segmentation measures the combined cadence instead.
            'sam': (not parallel_shared and self.config.get('processing_mode','classify_and_filter') == 'classify_and_filter'
                    and bool(self.config.get('mirror_analysis',False))),
            'mirror_masks': bool(self.config.get('mirror_analysis',False)) and not parallel_shared,
            'mirror_evidence': bool(self.config.get('mirror_analysis',False)),
            'segmentation': self.config.get('processing_mode','classify_and_filter') == 'classify_and_filter',
        }
        self.estimator = RunEstimator(applies=applies,profile=load_profile(self.timing_profile_path),parallel=parallel_shared)
        self.apply_inspection_counts()

    def command(self,command):
        self.process.write((json.dumps({'version':1,'command':command})+'\n').encode())
        self.phase.setText(tr('command_'+command))
        if command == 'pause':
            self.pause_button.setEnabled(False)
        if command == 'cancel':
            self.cancel_button.setEnabled(False)

    def resume(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.command('resume')
        elif self.config and not self.finished_ok:
            self.launch_worker()

    def resume_saved(self):
        directory = self.source.parent/'klassifiziert' if self.source else self.selection_directory()
        path,_ = QFileDialog.getOpenFileName(self,tr('resume_saved'),str(directory),'JSON (*.json)')
        if path:
            try:
                config = json.loads(Path(path).read_text(encoding='utf-8'))
                self.source = Path(config['source'])
                self.update_source_ui()
                if (Path(config['output'])/'report.json').exists():
                    self.config = config
                    self.finished_ok = True
                    self.set_running(False)
                    self.phase.setText(tr('run_done'))
                    self.open_file('report.html')
                    return
                self.config = config
                self.preset = json.loads(Path(config['preset']).read_text(encoding='utf-8'))
                processing_mode = config.get('processing_mode','classify_and_filter')
                self.processing_mode.setCurrentIndex(max(0,self.processing_mode.findData(processing_mode)))
                self.input_mode.setCurrentIndex(max(0,self.input_mode.findData(config.get('input_mode',DEFAULTS['input_mode']))))
                self.load_table()
                self.launch_worker()
            except Exception as error:
                self.log.appendPlainText(str(error))
                self.phase.setText(tr('launch_failed'))

    def read_stdout(self):
        self.pending += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.pending:
            line,self.pending = self.pending.split(b'\n',1)
            text = line.decode('utf-8',errors='replace')
            try:
                event = json.loads(text)
                if isinstance(event,dict) and event.get('version') == 1:
                    self.handle_event(event)
                else:
                    self.log.appendPlainText(text)
            except ValueError:
                self.log.appendPlainText(text)

    def read_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace').rstrip()
        if text:
            self.log.appendPlainText(text)

    def handle_event(self,event):
        kind = event['event']
        fallback = self.current_elapsed()
        timestamp = event.get('elapsed_seconds',fallback)
        if not isinstance(timestamp,(int,float)):
            timestamp = fallback
        else:
            self.elapsed_at_event = float(timestamp)
            self.elapsed_event_clock = time.monotonic()
        if kind == 'phase_started':
            started_phase = event['phase']
            self.phase_starts[started_phase] = event.get('elapsed_seconds',0)
            self.estimator.phase_started(started_phase,timestamp)
            # Only pipeline root phases have a meaningful duration.  Scene
            # detection is performed once per scan inside segmentation.
            if started_phase in ('inspection','sam','mirror','segmentation','voting','output'):
                self.estimate_phase = started_phase
                self.estimate_fraction = 0.
                self.phase_fraction = 0.
        if kind in ('phase_started','progress'):
            self.phase_key = event.get('phase','')
            if self.phase_key in ('mirror_masks','mirror_evidence'):
                self.phase_key = 'mirror'
            self.phase.setText(STRINGS.get('phase_'+event.get('phase',''),tr('state_running')))
        if kind == 'progress':
            fraction = max(0,min(1,event.get('fraction',0)))
            self.phase_fraction = fraction
            self.phase_progress.setValue(round(fraction*1000))
            phase = event.get('phase')
            estimate_phase = 'mirror' if phase in ('mirror_masks','mirror_evidence') else ('output' if phase == 'output_prepare' else phase)
            if estimate_phase in ('inspection','sam','mirror','segmentation','voting','output'):
                self.estimate_phase = estimate_phase
                self.estimate_fraction = (.1*fraction if phase == 'output_prepare' else (.1+.9*fraction if phase == 'output' else fraction))
            output_fraction = .1*fraction if phase == 'output_prepare' else .1+.9*fraction if phase == 'output' else 0.
            current_config = self.config or {}
            shared_sam = (current_config.get('processing_mode','classify_and_filter') == 'classify_and_filter'
                           and bool(current_config.get('mirror_analysis',False)))
            parallel_shared = shared_sam and bool(current_config.get('pipeline_parallel',DEFAULTS['pipeline_parallel']))
            if parallel_shared and phase in ('sam','mirror_masks','segmentation'):
                self.phase_key = 'parallel'
                self.phase.setText(STRINGS.get('phase_parallel',tr('state_running')))
            overall = (.1+.7*fraction if phase == 'segmentation' and parallel_shared
                       else .1+.35*fraction if phase in ('sam','mirror_masks') and parallel_shared
                       else .3*fraction if phase == 'sam' else .5+.4*fraction if phase == 'segmentation' and shared_sam
                        else .3+.2*fraction if phase in ('mirror_masks','mirror_evidence') and shared_sam
                        else .3+.6*fraction if phase == 'segmentation' else .9+.1*output_fraction if phase in ('output_prepare','output')
                        else .03+.27*fraction if phase in ('mirror_masks','mirror_evidence') else .03*fraction)
            self.current_fraction = max(self.current_fraction,overall)
            self.overall.setValue(round(self.current_fraction*1000))
            self.details.setText(tr('progress_detail',scan=event.get('scan','–'),scans=event.get('scan_count','–'),
                image=event.get('image','–'),images=event.get('image_count','–'),view=event.get('view','–'),views=event.get('view_count','–')))
            self.estimator.progress(phase,timestamp,**{key:value for key,value in event.items()
                                                       if key not in ('event','phase','elapsed_seconds')})
        elif kind == 'phase_finished':
            self.estimator.phase_finished(event.get('phase',''),timestamp)
            if event.get('phase') == 'inspection':
                self.read_inspection()
        elif kind == 'scene':
            self.scan_table.setItem(event['scan']-1,6,QTableWidgetItem(tr('scene_'+event['scene'])))
        elif kind == 'state':
            self.estimator.state(event['state'],timestamp)
            self.phase.setText(STRINGS.get('state_'+event['state'],tr('state_running')))
            self.resume_button.setEnabled(event['state'] == 'paused')
            self.pause_button.setEnabled(event['state'] == 'running')
        elif kind == 'input_mode':
            report = getattr(self, 'inspection_report', None)
            if report is not None:
                report['input_mode'] = {key: event[key] for key in ('mode', 'requested', 'rules', 'mixed', 'warnings') if key in event}
            self.show_input_mode(event)
        elif kind == 'warning':
            self.log.appendPlainText(STRINGS.get(event.get('code'),tr('technical_warning')))
        elif kind == 'cache_cleanup':
            self.cache_cleanup_freed = event.get('freed_bytes', 0)
        elif kind == 'error':
            if event.get('code') == 'input_rejected':
                self.input_rejected = True
                self.phase.setText(tr('input_rejected'))
                self.read_inspection()
                message = event.get('detail',tr('input_no_usable_images'))
                self.result_label.setText(message)
                self.result_label.setWordWrap(True)
                self.log.appendPlainText(message)
                warning = QMessageBox(QMessageBox.Icon.Warning,tr('input_rejected'),message,
                                      QMessageBox.StandardButton.Ok,self)
                warning.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                warning.open()
            else:
                self.phase.setText(tr('run_failed'))
        elif kind == 'finished':
            self.finished_ok = True
            self.current_fraction = 1
            self.overall.setValue(1000)
            self.phase.setText(tr('run_done'))
            freed = getattr(self, 'cache_cleanup_freed', 0)/1_000_000_000
            self.result_label.setText(tr('result_ready',path=event['output'])+f'\nZwischenspeicher freigegeben: {freed:.2f} GB')
            elapsed = int(event.get('elapsed_seconds',0))
            self.bottom.setText(tr('elapsed',duration=format_duration(elapsed,True))+' · '+tr('run_done'))
            if self.timing_profile_path:
                try:
                    save_profile(self.timing_profile_path,self.estimator)
                except Exception as error:
                    self.log.appendPlainText(str(error))
        if kind != 'progress':
            self.log.appendPlainText(json.dumps(event,ensure_ascii=False))

    def read_inspection(self,path=None):
        try:
            report_path = Path(path) if path is not None else Path(self.config['output'])/'inspect_report.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.set_summary_status(tr('input_summary',scans=report['scan_count'],images=report['image_count'],points=report['record_count']))
            self.scan_table.setRowCount(len(report['scans']))
            for row,scan in enumerate(report['scans']):
                meta = scan['metadata']
                images = [report['images'][i-1] for i in scan['image_indices']]
                kinds = sorted({r for image in images for r in image['representations']})
                values = [row+1,meta.get('name',''),scan['record_count'],len(images),', '.join(kinds),
                          tr('pose_'+scan['pose_status']),tr('scene_both'),tr('available') if scan['classification_eligible_metadata'] else tr('not_classifiable')]
                for col,value in enumerate(values):
                    self.scan_table.setItem(row,col,QTableWidgetItem(str(value)))
            self.scan_table.resizeColumnsToContents()
            self.metadata.setPlainText(json.dumps(report['metadata'],ensure_ascii=False,indent=2)[:30000])
            self.inspection_report = report
            self.show_input_mode(report.get('input_mode'))
            self.update_slam_ui()
            self.apply_inspection_counts()
            return True
        except Exception as error:
            self.log.appendPlainText(str(error))
            return False

    def show_input_mode(self, value):
        if not value or not hasattr(self, 'input_mode_status'):
            return
        mode = value.get('mode', 'static')
        title = tr('input_mode_slam_detected') if mode == 'slam' else tr('input_mode_static_detected')
        reasons = [rule.get('detail', '') for rule in value.get('rules', []) if rule.get('hit')]
        if not reasons:
            reasons = [tr('input_mode_no_slam_evidence')]
        text = title+'\n'+tr('input_mode_reasons', reasons='; '.join(reasons))
        if value.get('mixed'):
            text += '\n'+tr('input_mode_mixed')
        self.input_mode_status.setText(text)
        self.input_mode_status.setProperty('status', 'warning' if mode == 'slam' or value.get('mixed') else 'success')
        self.input_mode_status.style().unpolish(self.input_mode_status)
        self.input_mode_status.style().polish(self.input_mode_status)

    def apply_inspection_counts(self):
        report = getattr(self,'inspection_report',None)
        if report is None:
            return
        self.estimator.set_unit_count('mirror_masks',sum(bool(scan['image_indices']) for scan in report['scans']))
        self.estimator.set_unit_count('mirror_evidence',report['scan_count'])
        self.estimator.set_unit_count('sam',report['scan_count'])
        self.estimator.set_unit_count('segmentation',sum(bool(scan['classification_eligible_metadata']) for scan in report['scans']))
        self.estimator.set_unit_indices('mirror_evidence',range(1,report['scan_count']+1))
        self.estimator.set_unit_indices('sam',range(1,report['scan_count']+1))
        self.estimator.set_unit_indices('segmentation',[index for index,scan in enumerate(report['scans'],1)
                                                        if scan['classification_eligible_metadata']])

    def set_running(self,running):
        self.class_widget.setEnabled(not running and self.processing_mode.currentData() != 'filter_only')
        self.settings_widget.setEnabled(not running)
        self.select_button.setEnabled(not running)
        self.saved_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.source is not None)
        self.pause_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        self.resume_button.setEnabled(not running and self.config is not None and not self.finished_ok and not self.input_rejected)
        self.open_output.setEnabled(self.config is not None)
        self.open_log.setEnabled(self.config is not None)
        self.open_report.setEnabled(self.config is not None and (Path(self.config['output'])/'report.html').exists())
        if not running:
            self.validate()

    def worker_finished(self,code,status):
        self.read_stdout()
        self.read_stderr()
        self.set_running(False)
        if self.input_rejected:
            self.phase.setText(tr('input_rejected'))
        elif code == 130:
            self.phase.setText(tr('state_cancelled'))
        elif code != 0 or not self.finished_ok:
            self.phase.setText(tr('run_failed'))
        if self.closing:
            self.close()

    def process_error(self,error):
        self.log.appendPlainText(self.process.errorString())
        self.phase.setText(tr('run_failed'))
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.set_running(False)

    def update_clock(self):
        if self.finished_ok or not self.started or (not self.watching and self.process.state() == QProcess.ProcessState.NotRunning):
            return
        elapsed = self.current_elapsed()
        remaining = self.estimator.remaining(elapsed)
        self.bottom.setText(tr('elapsed',duration=format_duration(elapsed,True))+' · '+
                            (tr('remaining',duration=format_duration(remaining[2]),max_duration=format_duration(remaining[3])) if remaining is not None else tr('estimate_pending'))+' · '+
                            (tr('total_remaining',duration=format_duration(remaining[0]),max_duration=format_duration(remaining[1])) if remaining is not None else tr('total_estimate_pending')))

    def current_elapsed(self):
        if self.elapsed_at_event is not None and self.elapsed_event_clock is not None:
            return max(0.,self.elapsed_at_event+time.monotonic()-self.elapsed_event_clock)
        return max(0.,time.monotonic()-self.started) if self.started else 0.

    def open_file(self,name):
        if self.config:
            path = Path(self.config['output'])/name
            if path.exists():
                os.startfile(str(path))

    def closeEvent(self,event):
        self.stop_preview()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.closing = True
            self.command('cancel')
            event.ignore()
        else:
            event.accept()
