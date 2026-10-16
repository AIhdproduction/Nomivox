"""Shared dark application theme for the Nomivox desktop interface."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication


TOKENS = {
    "background": "#0B1120",
    "surface": "#111827",
    "surface_raised": "#1E293B",
    "border": "#334155",
    "text": "#E2E8F0",
    "text_muted": "#A7B3C6",
    "primary": "#3B82F6",
    "primary_hover": "#60A5FA",
    "primary_pressed": "#2563EB",
    "on_primary": "#FFFFFF",
    "accent": "#F59E0B",
    "success": "#22C55E",
    "danger": "#EF4444",
    "focus": "#60A5FA",
    "selection": "#1D4ED8",
    "disabled": "#64748B",
}


def _checkbox_icons(tokens):
    """Create compact check marks for QSS-controlled check indicators."""
    directory = Path(__file__).resolve().parents[2] / ".cache" / "temp"
    directory.mkdir(parents=True, exist_ok=True)
    icons = {}
    for name, color in (("checked", tokens["on_primary"]), ("checked_disabled", tokens["disabled"])):
        path = directory / f"nomivox_{name}.png"
        if not path.is_file():
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                       Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(3, 8, 7, 12)
            painter.drawLine(7, 12, 13, 4)
            painter.end()
            pixmap.save(str(path))
        icons[name] = path.as_posix()
    return icons


def _stylesheet(tokens, icons):
    return """
    * {{
        color: {text};
        font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
        font-size: 10pt;
    }}
    QMainWindow, QWidget#appRoot {{ background: {background}; }}
    QWidget {{ background: transparent; }}
    QFrame#headerBar {{ background: {surface}; border-bottom: 1px solid {border}; }}
    QLabel#wordmark {{ font-size: 18pt; font-weight: 600; }}
    QLabel#subtitle, QLabel#currentFile, QLabel[class="hint"] {{ color: {text_muted}; }}
    QLabel#cardHeading {{ font-size: 11pt; font-weight: 600; }}
    QFrame[class="card"] {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
    QFrame#sidebar {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; }}
    QListWidget#navigation {{ background: transparent; border: 0; outline: 0; padding: 4px; }}
    QListWidget#navigation::item {{ border-radius: 6px; padding: 10px 12px; margin: 2px; color: {text_muted}; }}
    QListWidget#navigation::item:hover {{ background: {surface_raised}; color: {text}; }}
    QListWidget#navigation::item:selected {{ background: {selection}; color: {on_primary}; font-weight: 600; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {{
        background: {surface_raised}; border: 1px solid {border}; border-radius: 8px; padding: 6px 8px;
        selection-background-color: {selection}; selection-color: {on_primary};
    }}
    QComboBox::drop-down {{ border: 0; width: 24px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{ border: 0; width: 18px; }}
    QPushButton, QToolButton {{ background: {surface_raised}; border: 1px solid {border}; border-radius: 8px; padding: 7px 12px; min-height: 18px; }}
    QPushButton:hover, QToolButton:hover {{ background: {border}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {background}; }}
    QPushButton:disabled, QToolButton:disabled {{ background: {surface}; border-color: {surface}; color: {disabled}; }}
    QPushButton[role="primary"] {{ background: {primary}; border-color: {primary}; color: {on_primary}; font-weight: 600; }}
    QPushButton[role="primary"]:hover {{ background: {primary_hover}; border-color: {primary_hover}; }}
    QPushButton[role="primary"]:pressed {{ background: {primary_pressed}; border-color: {primary_pressed}; }}
    QPushButton[role="danger"] {{ border-color: {danger}; color: {danger}; }}
    QPushButton[role="danger"]:hover {{ background: {danger}; color: {on_primary}; }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QCheckBox:disabled {{ color: {disabled}; }}
    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {border}; border-radius: 4px; background: {surface_raised}; }}
    QCheckBox::indicator:hover {{ border-color: {focus}; }}
    QCheckBox::indicator:focus {{ border: 2px solid {focus}; }}
    QCheckBox::indicator:checked {{ background: {primary}; border-color: {primary}; image: url({checked}); }}
    QCheckBox::indicator:checked:hover {{ background: {primary_hover}; border-color: {primary_hover}; }}
    QCheckBox::indicator:checked:disabled {{ background: {surface}; border-color: {border}; image: url({checked_disabled}); }}
    QTableWidget::indicator {{ width: 16px; height: 16px; border: 1px solid {border}; border-radius: 4px; background: {surface_raised}; }}
    QTableWidget::indicator:hover {{ border-color: {focus}; }}
    QTableWidget::indicator:checked {{ background: {primary}; border-color: {primary}; image: url({checked}); }}
    QTableWidget::indicator:checked:disabled {{ background: {surface}; border-color: {border}; image: url({checked_disabled}); }}
    QProgressBar {{ background: {surface_raised}; border: 1px solid {border}; border-radius: 5px; min-height: 8px; text-align: center; }}
    QProgressBar::chunk {{ background: {primary}; border-radius: 4px; }}
    QHeaderView::section {{ background: {surface_raised}; border: 0; border-bottom: 1px solid {border}; padding: 7px; font-weight: 600; }}
    QTableWidget {{ gridline-color: {border}; alternate-background-color: {surface}; }}
    QTableWidget::item {{ padding: 4px; }}
    QTableWidget::item:selected {{ background: {selection}; color: {on_primary}; }}
    QScrollBar:vertical {{ background: {surface}; width: 10px; margin: 2px 0; border: 0; }}
    QScrollBar::handle:vertical {{ background: {border}; min-height: 24px; border-radius: 5px; }}
    QScrollBar::handle:vertical:hover {{ background: {text_muted}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QScrollBar:horizontal {{ background: {surface}; height: 10px; margin: 0 2px; border: 0; }}
    QScrollBar::handle:horizontal {{ background: {border}; min-width: 24px; border-radius: 5px; }}
    QScrollBar::handle:horizontal:hover {{ background: {text_muted}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
    QPlainTextEdit#logView, QPlainTextEdit#metadataView {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; }}
    QLabel[status="success"] {{ color: {success}; }}
    QLabel[status="warning"] {{ color: {accent}; }}
    QLabel[status="danger"] {{ color: {danger}; }}
    *:focus {{ outline: 2px solid {focus}; outline-offset: 1px; }}
    """.format(**tokens, **icons)


def apply_theme(app: QApplication | None = None):
    """Apply the shared Fusion palette and stylesheet; safe to call repeatedly."""
    app = app or QApplication.instance()
    if app is None:
        return
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(TOKENS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(TOKENS["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(TOKENS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(TOKENS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TOKENS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(TOKENS["selection"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TOKENS["on_primary"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TOKENS["text_muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TOKENS["text_muted"]))
    app.setPalette(palette)
    app.setStyleSheet(_stylesheet(TOKENS, _checkbox_icons(TOKENS)))
