"""German Qt interface; expensive operations execute in a separate QProcess."""
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,
    QLabel,QLineEdit,QPushButton,QComboBox,QTableWidget,QTableWidgetItem,QCheckBox,QPlainTextEdit,
    QProgressBar,QFileDialog,QMessageBox,QDoubleSpinBox,QSpinBox,QScrollArea,QInputDialog)
from e57cls.files import atomic_json
from e57cls.pipeline.config import PROJECT,DEFAULTS,resolve_config
from e57cls.scene.presets import validate_preset

STRINGS = json.loads((PROJECT/'e57cls/gui/strings_de.json').read_text(encoding='utf-8'))
def tr(key, **values):
    return STRINGS[key].format(**values)


class MainWindow(QMainWindow):
    def __init__(self, source, auto_start=True):
        super().__init__()
        self.source = Path(source)
        self.preset = json.loads((PROJECT/'presets/default.json').read_text(encoding='utf-8'))
        self.config = None
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.worker_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.pending = b''
        self.started = 0
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
        self.setWindowTitle(tr('app_title'))
        self.resize(1180,800)
        root = QWidget()
        layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.input_tab()
        self.classes_tab()
        self.settings_tab()
        self.run_tab()
        self.bottom = QLabel(tr('ready'))
        self.overall = QProgressBar()
        self.overall.setRange(0,1000)
        layout.addWidget(self.overall)
        layout.addWidget(self.bottom)
        self.setCentralWidget(root)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.load_table()
        self.set_running(False)
        if auto_start:
            QTimer.singleShot(100,self.start_run)

    def tab(self,title):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.tabs.addTab(widget,tr(title))
        return widget,layout

    def button(self,key,callback,layout):
        button = QPushButton(tr(key))
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def input_tab(self):
        _,layout = self.tab('tab_input')
        self.input_path = QLineEdit(str(self.source))
        self.input_path.setReadOnly(True)
        layout.addWidget(self.input_path)
        self.select_button = self.button('select_another',self.select_source,layout)
        hint = QLabel(tr('automatic_flow'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.summary = QLabel(tr('summary_pending'))
        layout.addWidget(self.summary)
        self.scan_table = QTableWidget(0,8)
        self.scan_table.setHorizontalHeaderLabels([tr(k) for k in ('scan','name','point_count','images','representation','pose','scene','status')])
        self.scan_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scan_table.cellDoubleClicked.connect(self.override_scene)
        layout.addWidget(self.scan_table)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setMaximumHeight(150)
        layout.addWidget(self.metadata)

    def classes_tab(self):
        self.class_widget,layout = self.tab('tab_classes')
        row = QHBoxLayout()
        row.addWidget(QLabel(tr('scene')))
        self.scene_mode = QComboBox()
        for mode in ('both','indoor','outdoor'):
            self.scene_mode.addItem(tr('scene_'+mode),mode)
        row.addWidget(self.scene_mode)
        self.scene_mode.currentIndexChanged.connect(self.filter_classes)
        for key,callback in [('load_preset',self.load_preset),('save_preset',self.save_preset),('duplicate_preset',self.save_preset),('reset_preset',self.reset_preset)]:
            self.button(key,callback,row)
        layout.addLayout(row)
        self.class_table = QTableWidget(0,9)
        self.class_table.setHorizontalHeaderLabels([tr(k) for k in ('enabled','name','internal_name','code','color','prompts','scene','instance_threshold','mask_threshold')])
        self.class_table.itemChanged.connect(self.edit_class)
        layout.addWidget(self.class_table)
        buttons = QHBoxLayout()
        self.button('add_class',self.add_class,buttons)
        self.button('remove_class',self.remove_class,buttons)
        layout.addLayout(buttons)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        self.mirror = QCheckBox(tr('mirror_enable'))
        self.mirror.setChecked(DEFAULTS['mirror_analysis'])
        self.glass = QCheckBox(tr('glass_enable'))
        self.glass.setChecked(DEFAULTS['glass_analysis'])
        self.extra = QCheckBox(tr('extra_no_reflection'))
        layout.addWidget(self.mirror)
        layout.addWidget(self.glass)
        layout.addWidget(self.extra)
        hint = QLabel(tr('mirror_limits')+' '+tr('mirror_preservation'))
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def settings_tab(self):
        self.settings_widget,layout = self.tab('tab_settings')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        form = QFormLayout(contents)
        scroll.setWidget(contents)
        layout.addWidget(scroll)
        self.profile = QComboBox()
        for key in ('fast','balanced','high'):
            self.profile.addItem(tr('profile_'+key),key)
        self.profile.setCurrentIndex(1)
        form.addRow(tr('profile'),self.profile)
        self.controls = {}
        specifications = {
            'image_scale':(.1,1.,.05),'tile_overlap':(0.,.8,.05),'min_observations':(1,100,1),
            'voxel_size':(.001,1.,.005),'tile_size':(336,1512,16),'prompt_batch':(1,64,1),
            'instance_confidence':(0.,1.,.05),'mask_confidence':(0.,1.,.05),'boundary_margin':(0,16,1),
            'min_winning_share':(0.,1.,.05),'view_size':(64,1512,16),'view_fov':(30,120,5),
            'visibility_abs':(.001,.5,.01),'visibility_rel':(0.,.1,.005),'depth_width':(256,8192,256),
            'gpu_memory_fraction':(.1,1.,.05),'scene_sky_outdoor':(0.,1.,.01),
            'scene_ceiling_outdoor_max':(0.,1.,.01),'scene_ceiling_indoor':(0.,1.,.01),
            'scene_sky_indoor_max':(0.,1.,.01),'scene_overhead_range':(1.,30.,1.),
            'scene_plane_tolerance':(.001,.1,.005),'scene_plane_fraction':(.1,1.,.05),
            'scene_ceiling_supported':(0.,1.,.01),'scene_upper_returns_indoor':(0.,1.,.05),
            'scene_open_sky_outdoor':(0.,1.,.01),'scene_open_sky_indoor_max':(0.,1.,.01),
            'scene_sky_confidence':(0.,1.,.05),'scene_ceiling_confidence':(0.,1.,.05),
            'stairs_min_confidence':(0.,1.,.05),
            'mirror_mask_confidence':(0.,1.,.05),'mirror_fit_tolerance':(.001,.1,.005),
            'mirror_plane_tolerance':(.001,.2,.005),'reflection_prompt_agreement':(1,8,1),
            'mirror_merge_angle':(.1,10.,.5),'mirror_merge_distance':(.001,.1,.005),
            'mirror_max_range':(1.,100.,1.),'mirror_min_pixels':(10,10000,10),'mirror_mask_width':(512,4096,128),
            'mirror_frame_band':(1,60,1),'mirror_frame_band_max':(1,60,1)}
        for key,(low,high,step) in specifications.items():
            widget = QSpinBox() if isinstance(low,int) else QDoubleSpinBox()
            if isinstance(widget,QDoubleSpinBox):
                widget.setDecimals(3)
            widget.setRange(low,high)
            widget.setSingleStep(step)
            widget.setValue(DEFAULTS[key])
            self.controls[key] = widget
            form.addRow(tr('setting_'+key),widget)
        self.profile.currentIndexChanged.connect(self.apply_profile)
        self.output_format = QComboBox()
        for key in ('las','laz'):
            self.output_format.addItem(key.upper()+' 1.4',key)
        form.addRow(tr('output_format'),self.output_format)
        self.diagnostics = QComboBox()
        for key in ('off','sample','all'):
            self.diagnostics.addItem(tr('diagnostics_'+key),key)
        self.diagnostics.setCurrentIndex(1)
        form.addRow(tr('diagnostics'),self.diagnostics)
        self.cache_path = QLineEdit(str(PROJECT/'.cache'))
        self.gpu = QLineEdit('cuda:0')
        form.addRow(tr('cache_folder'),self.cache_path)
        form.addRow(tr('gpu_device'),self.gpu)
        self.strict = QCheckBox(tr('strict_mode'))
        form.addRow(self.strict)
        self.stairs_priority = QCheckBox(tr('stairs_priority'))
        self.stairs_priority.setChecked(DEFAULTS['stairs_priority'])
        form.addRow(self.stairs_priority)

    def run_tab(self):
        _,layout = self.tab('tab_run')
        buttons = QHBoxLayout()
        self.start_button = self.button('start',self.start_run,buttons)
        self.pause_button = self.button('pause',lambda:self.command('pause'),buttons)
        self.cancel_button = self.button('cancel',lambda:self.command('cancel'),buttons)
        self.resume_button = self.button('resume',self.resume,buttons)
        self.saved_button = self.button('resume_saved',self.resume_saved,buttons)
        layout.addLayout(buttons)
        self.phase = QLabel(tr('ready'))
        self.details = QLabel()
        self.phase_progress = QProgressBar()
        self.phase_progress.setRange(0,1000)
        layout.addWidget(self.phase)
        layout.addWidget(self.details)
        layout.addWidget(self.phase_progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        layout.addWidget(self.log)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        buttons = QHBoxLayout()
        self.open_output = self.button('open_output',lambda:self.open_file(''),buttons)
        self.open_report = self.button('open_report',lambda:self.open_file('report.html'),buttons)
        self.open_log = self.button('open_log',lambda:self.open_file('log.txt'),buttons)
        layout.addLayout(buttons)

    def watch(self,output):
        self.config = json.loads((Path(output)/'run.json').read_text(encoding='utf-8'))
        self.watching = True
        self.watch_offset = 0
        self.preset = json.loads(Path(self.config['preset']).read_text(encoding='utf-8'))
        self.load_table()
        for key,widget in self.controls.items():
            widget.setValue(self.config.get(key,DEFAULTS[key]))
        self.mirror.setChecked(self.config.get('mirror_analysis',False))
        self.glass.setChecked(self.config.get('glass_analysis',False))
        self.stairs_priority.setChecked(self.config.get('stairs_priority',False))
        self.scene_mode.setCurrentIndex(self.scene_mode.findData(self.config['scene_mode']))
        self.read_inspection()
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
                self.class_table.setItem(row,col,QTableWidgetItem(value))
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
        self.validate()

    def validate(self):
        errors,warnings = validate_preset(self.preset)
        messages = [tr('validation_'+e.split(':')[0])+(': '+e.split(':',1)[1] if ':' in e else '') for e in errors+warnings]
        self.validation.setText('\n'.join(messages) if messages else tr('preset_valid'))
        if hasattr(self,'start_button'):
            self.start_button.setEnabled(not errors and self.process.state() == QProcess.ProcessState.NotRunning)
        return not errors

    def add_class(self):
        code = next((c for c in range(64,256) if c not in {x['code'] for x in self.preset['classes']}|{68}),64)
        self.preset['classes'].append(dict(id='class_'+uuid4().hex[:6],name_de=tr('new_class'),name_en='New class',code=code,
            color='#888888',scene='both',enabled=False,prompts=['object'],min_instance_conf=None,min_mask_conf=None))
        self.load_table()

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
        scale,overlap,observations = {'fast':(.5,.1,1),'balanced':(.75,.2,1),'high':(1.,.3,2)}[self.profile.currentData()]
        for key,value in [('image_scale',scale),('tile_overlap',overlap),('min_observations',observations)]:
            self.controls[key].setValue(value)

    def select_source(self):
        path,_ = QFileDialog.getOpenFileName(self,tr('select_input'),str(self.source.parent),'E57 (*.e57)')
        if path:
            self.source = Path(path)
            self.input_path.setText(path)
            self.config = None
            self.overrides = {}
            self.start_run()

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
        if self.watching or self.process.state() != QProcess.ProcessState.NotRunning or not self.validate():
            return
        try:
            settings = {key:widget.value() for key,widget in self.controls.items()}
            settings.update(source=str(self.source),scene_mode=self.scene_mode.currentData(),profile=self.profile.currentData(),
                mirror_analysis=self.mirror.isChecked(),glass_analysis=self.glass.isChecked(),
                stairs_priority=self.stairs_priority.isChecked(),extra_no_reflection=self.extra.isChecked(),format=self.output_format.currentData(),
                diagnostics=self.diagnostics.currentData(),cache=self.cache_path.text(),gpu_device=self.gpu.text(),strict=self.strict.isChecked(),
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
            atomic_json(PROJECT/'.cache/launcher.json',{'last_folder':str(self.source.parent),'last_config':str(output/'run.json')})
            self.launch_worker()
        except Exception as error:
            self.log.appendPlainText(str(error))
            self.phase.setText(tr('launch_failed'))

    def launch_worker(self):
        import sys
        self.pending = b''
        self.started = time.monotonic()
        self.current_fraction = 0
        self.phase_starts = {}
        self.phase_fraction = 0.
        self.phase_key = ''
        self.finished_ok = False
        self.input_rejected = False
        env = QProcessEnvironment.systemEnvironment()
        for key,value in {'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1',
                          'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TEMP':str(Path(self.config['cache'])/'temp'),
                          'TMP':str(Path(self.config['cache'])/'temp')}.items():
            env.insert(key,value)
        Path(self.config['cache'],'temp').mkdir(parents=True,exist_ok=True)
        self.process.setProcessEnvironment(env)
        self.set_running(True)
        self.phase.setText(tr('state_running'))
        self.result_label.setText(tr('result_location',path=self.config['output']))
        self.tabs.setCurrentIndex(3)
        self.process.start(sys.executable,['-m','e57cls','run','--config',str(Path(self.config['output'])/'run.json')])

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
        path,_ = QFileDialog.getOpenFileName(self,tr('resume_saved'),str(self.source.parent/'klassifiziert'),'JSON (*.json)')
        if path:
            try:
                config = json.loads(Path(path).read_text(encoding='utf-8'))
                if (Path(config['output'])/'report.json').exists():
                    self.config = config
                    self.finished_ok = True
                    self.set_running(False)
                    self.phase.setText(tr('run_done'))
                    self.open_file('report.html')
                    return
                self.config = config
                self.source = Path(config['source'])
                self.input_path.setText(str(self.source))
                self.preset = json.loads(Path(config['preset']).read_text(encoding='utf-8'))
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
        if kind == 'phase_started':
            self.phase_starts[event['phase']] = event.get('elapsed_seconds',0)
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
            overall = (.3+.6*fraction if phase == 'segmentation' else .9+.1*fraction if phase == 'output'
                       else .03+.27*fraction if phase in ('mirror_masks','mirror_evidence') else .03*fraction)
            self.current_fraction = max(self.current_fraction,overall)
            self.overall.setValue(round(self.current_fraction*1000))
            self.details.setText(tr('progress_detail',scan=event.get('scan','–'),scans=event.get('scan_count','–'),
                image=event.get('image','–'),images=event.get('image_count','–'),view=event.get('view','–'),views=event.get('view_count','–')))
        elif kind == 'phase_finished' and event.get('phase') == 'inspection':
            self.read_inspection()
        elif kind == 'scene':
            self.scan_table.setItem(event['scan']-1,6,QTableWidgetItem(tr('scene_'+event['scene'])))
        elif kind == 'state':
            self.phase.setText(STRINGS.get('state_'+event['state'],tr('state_running')))
            self.resume_button.setEnabled(event['state'] == 'paused')
            self.pause_button.setEnabled(event['state'] == 'running')
        elif kind == 'warning':
            self.log.appendPlainText(STRINGS.get(event.get('code'),tr('technical_warning')))
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
            self.result_label.setText(tr('result_ready',path=event['output']))
            elapsed = int(event.get('elapsed_seconds',0))
            self.bottom.setText(tr('elapsed',minutes=elapsed//60,seconds=elapsed%60)+' · '+tr('run_done'))
        if kind != 'progress':
            self.log.appendPlainText(json.dumps(event,ensure_ascii=False))

    def read_inspection(self):
        try:
            report = json.loads((Path(self.config['output'])/'inspect_report.json').read_text(encoding='utf-8'))
            self.summary.setText(tr('input_summary',scans=report['scan_count'],images=report['image_count'],points=report['record_count']))
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
        except Exception as error:
            self.log.appendPlainText(str(error))

    def set_running(self,running):
        self.class_widget.setEnabled(not running)
        self.settings_widget.setEnabled(not running)
        self.select_button.setEnabled(not running)
        self.saved_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.cancel_button.setEnabled(running)
        self.resume_button.setEnabled(not running and self.config is not None and not self.finished_ok and not self.input_rejected)
        self.open_output.setEnabled(self.config is not None)
        self.open_log.setEnabled(self.config is not None)
        self.open_report.setEnabled(self.config is not None and (Path(self.config['output'])/'report.html').exists())

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
        if not self.started or (not self.watching and self.process.state() == QProcess.ProcessState.NotRunning):
            return
        elapsed = max(0,int(time.monotonic()-self.started))
        import math
        beginning = self.phase_starts.get(self.phase_key)
        phase_elapsed = max(0,elapsed-beginning) if beginning is not None else 0
        remaining = (max(1,math.ceil(phase_elapsed*(1-self.phase_fraction)/self.phase_fraction/60))
                     if beginning is not None and phase_elapsed >= 5 and .01 < self.phase_fraction < 1 else None)
        self.bottom.setText(tr('elapsed',minutes=elapsed//60,seconds=elapsed%60)+' · '+
                            (tr('remaining',minutes=remaining) if remaining is not None else tr('estimate_pending')))

    def open_file(self,name):
        if self.config:
            path = Path(self.config['output'])/name
            if path.exists():
                os.startfile(str(path))

    def closeEvent(self,event):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.closing = True
            self.command('cancel')
            event.ignore()
        else:
            event.accept()
