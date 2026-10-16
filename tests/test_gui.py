import json
import sys
import time
import pytest
from pathlib import Path
from PySide6.QtCore import QProcess,QCoreApplication,QEvent
from PySide6.QtWidgets import QApplication,QFileDialog,QInputDialog,QMessageBox
from e57cls.gui.eta import RunEstimator
from e57cls.gui import window as window_module
from e57cls.gui.window import MainWindow,tr


@pytest.fixture(scope='session')
def qt_application():
    app = QApplication.instance() or QApplication([])
    yield app
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_rejected_input_shows_warning_and_scan_counts(tmp_path, qt_application, monkeypatch):
    report = dict(scan_count=1, image_count=0, record_count=123, images=[], metadata={},
                  scans=[dict(metadata={'name':'without images'}, record_count=123,
                              image_indices=[], pose_status='valid', classification_eligible_metadata=False)])
    (tmp_path/'inspect_report.json').write_text(json.dumps(report), encoding='utf-8')
    window = MainWindow(tmp_path/'input.e57', auto_start=False)
    window.config = {'output':str(tmp_path)}
    messages = []
    monkeypatch.setattr(QMessageBox, 'open', lambda box: messages.append(box.text()))
    try:
        detail = tr('input_no_images')
        window.handle_event(dict(event='error', code='input_rejected', detail=detail))
        window.worker_finished(1,QProcess.ExitStatus.NormalExit)
        assert messages == [detail]
        assert window.phase.text() == tr('input_rejected')
        assert window.result_label.text() == detail
        assert window.scan_table.item(0,2).text() == '123'
        assert window.scan_table.item(0,3).text() == '0'
        assert window.scan_table.item(0,7).text() == tr('not_classifiable')
        assert "'points'" not in window.log.toPlainText()
        assert not window.resume_button.isEnabled()
        assert window.select_button.isEnabled()
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_window_without_source_allows_presets_and_later_selection(tmp_path, qt_application, monkeypatch):
    writes = []
    monkeypatch.setattr(window_module, 'atomic_json', lambda path,value: writes.append((Path(path),value)))
    window = MainWindow(None,auto_start=False)
    try:
        assert not window.start_button.isEnabled()
        assert not window.no_source_hint.isHidden()
        assert window.current_file.text() == tr('no_source_file')
        assert window.bottom.text() == tr('no_source')
        monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Testvorlage',True))
        window.save_preset()
        assert any(path.name.startswith('preset_') for path,_ in writes)
        source = tmp_path/'later.e57'
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(source),'E57 (*.e57)'))
        window.select_source()
        assert window.start_button.isEnabled()
        assert window.current_file.text() == source.name
        assert window.no_source_hint.isHidden()
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_reflection_eta_uses_parent_phase_and_measured_segmentation_rate(qt_application):
    app = qt_application
    window = MainWindow(Path('test.e57'),auto_start=False)
    try:
        window.watching = True
        window.started = time.monotonic()-600
        window.estimator = RunEstimator(applies={'segmentation': True},unit_counts={'segmentation': 4})
        window.handle_event(dict(event='phase_started',phase='segmentation',elapsed_seconds=400))
        window.handle_event(dict(event='phase_started',phase='scene',elapsed_seconds=500,scan=2))
        window.handle_event(dict(event='progress',phase='segmentation',fraction=.5,scan=1,scan_count=4,elapsed_seconds=600))
        window.update_clock()
        assert window.estimate_phase == 'segmentation'
        assert tr('remaining',duration='15 min',max_duration='28 min') in window.bottom.text()
        assert tr('total_remaining',duration='17 min',max_duration='32 min') in window.bottom.text()
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_total_eta_covers_phase_and_keeps_finished_status(qt_application, monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 10000.)
    window = MainWindow(Path('test.e57'),auto_start=False)
    try:
        window.watching = True
        window.started = 9400.
        window.estimator = RunEstimator(applies={'mirror_masks': False,'mirror_evidence': False,'segmentation': False})
        window.handle_event(dict(event='phase_started',phase='output',elapsed_seconds=0))
        window.handle_event(dict(event='progress',phase='output',fraction=.5,elapsed_seconds=600))
        window.update_clock()
        # A late slow phase must not have a shorter total ETA than its own ETA.
        assert tr('remaining',duration='10 min',max_duration='10 min') in window.bottom.text()
        assert tr('total_remaining',duration='10 min',max_duration='10 min') in window.bottom.text()
        window.handle_event(dict(event='finished',elapsed_seconds=600,output='test.las'))
        completed = window.bottom.text()
        window.update_clock()
        assert window.bottom.text() == completed
        assert tr('run_done') in completed
        assert 'Gesamtrestzeit' not in completed
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_gui_worker_pause_resume_cancel_and_responsive_timer(tmp_path,qt_application):
    app = qt_application
    window = MainWindow(tmp_path/'input.e57',auto_start=False)
    window.config = {'output':str(tmp_path)}
    window.started = time.monotonic()
    window.set_running(True)
    worker = '''
import sys,time
from pathlib import Path
from e57cls.pipeline.control import Control,Cancelled
control=Control(Path(sys.argv[1]))
try:
    for i in range(200):
        control.checkpoint()
        control.event('progress',phase='segmentation',fraction=i/200)
        time.sleep(.03)
except Cancelled:
    sys.exit(130)
finally:
    control.close()
'''
    def wait(predicate):
        deadline = time.monotonic()+8
        while not predicate() and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate()
    try:
        window.process.start(sys.executable,['-u','-c',worker,str(tmp_path)])
        wait(lambda:window.current_fraction>.05)
        window.command('pause')
        wait(lambda:window.phase.text()==tr('state_paused'))
        assert window.resume_button.isEnabled()
        window.resume()
        wait(lambda:window.pause_button.isEnabled())
        window.command('cancel')
        wait(lambda:window.process.state()==QProcess.ProcessState.NotRunning)
        assert window.phase.text()==tr('state_cancelled')
        assert window.resume_button.isEnabled()
        events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
        states=[e.get('state') for e in events]
        assert all(s in states for s in ('paused','running','cancelling','cancelled'))
        assert window.tabs.count()==4
    finally:
        if window.process.state()!=QProcess.ProcessState.NotRunning:
            window.process.kill()
            window.process.waitForFinished(5000)
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
