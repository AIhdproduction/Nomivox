import sys

from e57cls import launcher


def test_cancelled_file_dialog_opens_window_without_source(monkeypatch):
    sources = []

    class Window:
        def show(self):
            pass

    def build(source, auto_start=True):
        sources.append((source,auto_start))
        return Window()

    monkeypatch.setattr(sys, 'argv', ['start_classifier.bat'])
    monkeypatch.setattr(launcher.QFileDialog, 'getOpenFileName', lambda *args: ('',''))
    monkeypatch.setattr(launcher, 'build_window', build)
    monkeypatch.setattr(launcher.QApplication, 'exec', lambda app: 0)
    assert launcher.main() == 0
    assert sources == [(None,True)]
