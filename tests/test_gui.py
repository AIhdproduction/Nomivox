import json
import sys
import time
import pytest
from pathlib import Path
from PySide6.QtCore import QProcess,QCoreApplication,QEvent
from PySide6.QtWidgets import QApplication
from e57cls.gui.window import MainWindow,tr


@pytest.fixture(scope='session')
def qt_application():
    app = QApplication.instance() or QApplication([])
    yield app
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)


def test_reflection_eta_uses_parent_phase_and_never_shows_zero(qt_application):
    app = qt_application
    window = MainWindow(Path('test.e57'),auto_start=False)
    try:
        window.watching = True
        window.started = time.monotonic()-600
        window.handle_event(dict(event='phase_started',phase='mirror',elapsed_seconds=0))
        window.update_clock()
        assert tr('estimate_pending') in window.bottom.text()
        window.handle_event(dict(event='progress',phase='mirror_masks',fraction=.6,image=6,image_count=10))
        window.update_clock()
        assert window.phase_key == 'mirror'
        assert tr('remaining',minutes=7) in window.bottom.text()
        window.handle_event(dict(event='progress',phase='mirror_masks',fraction=.7,image=7,image_count=10))
        window.update_clock()
        assert tr('remaining',minutes=5) in window.bottom.text()
        window.handle_event(dict(event='phase_started',phase='segmentation',elapsed_seconds=600))
        window.update_clock()
        assert tr('estimate_pending') in window.bottom.text()
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
