"""Native E57 selection followed by an explicit workflow configuration."""
import json
from pathlib import Path
import sys


def main():
    from PySide6.QtWidgets import QApplication,QFileDialog
    from e57cls.gui.theme import apply_theme
    from e57cls.pipeline.config import PROJECT
    from e57cls.gui.window import MainWindow,tr
    from e57cls.files import atomic_json
    app = QApplication(sys.argv)
    apply_theme(app)
    if len(sys.argv) == 3 and sys.argv[1] == '--watch':
        output = Path(sys.argv[2])
        config = json.loads((output/'run.json').read_text(encoding='utf-8'))
        window = MainWindow(config['source'], auto_start=False)
        window.watch(output)
        window.show()
        return app.exec()
    if sys.argv[1:] == ['--check-startup']:
        import torch
        from importlib.metadata import version
        model = PROJECT/'models/sam.pt'
        if not model.is_file():
            model = PROJECT/'models/sam3.pt'
        print(json.dumps({'python':sys.version.split()[0],'qt':version('PySide6'),
                          'cuda':torch.cuda.is_available(),'model_path':str(model),
                          'model_present':model.is_file(),'available_stage':'classification','automatic_start':False}))
        return 0
    state_path = PROJECT/'.cache/launcher.json'
    state = {}
    try:
        state = json.loads(state_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        pass
    selected = sys.argv[1] if len(sys.argv)>1 else QFileDialog.getOpenFileName(None,tr('select_input'),
                 state.get('last_folder',str(PROJECT)),'E57 (*.e57)')[0]
    if not selected:
        return 0
    source = Path(selected).resolve()
    atomic_json(state_path,{**state,'last_folder':str(source.parent)})
    window = MainWindow(source)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
