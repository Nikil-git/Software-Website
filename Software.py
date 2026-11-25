import os
import sys
import time
import wave
import shutil
import threading
import subprocess
from datetime import datetime

import numpy as np
import cv2
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QEvent, pyqtSignal, QSettings
from PyQt6.QtGui import (QImage, QPainter, QPen, QColor, QAction, QKeySequence,
                         QPointingDevice, QPainterPath, QRadialGradient, QBrush)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QToolBar,
                             QPushButton, QLabel, QSlider, QFileDialog,
                             QDialog, QFormLayout, QComboBox, QCheckBox,
                             QDialogButtonBox, QVBoxLayout)

try:
    import sounddevice as sd
except Exception:
    sd = None

# ----------------------------------------------------------------- settings
CANVAS_W, CANVAS_H = 1920, 1080
FPS = 30
AUDIO_RATE = 44100
OUT_DIR = os.path.join(os.path.expanduser("~"), "Videos", "PhysicsBoard")
MAX_UNDO = 15
CAM_RES = 360
CAM_SIZES = [200, 280, 380]
CAM_CORNERS = ["bottom-right", "bottom-left", "top-right", "top-left"]
COLORS = {
    "White": "#ffffff", "Yellow": "#ffe14d", "Cyan": "#4dd9ff",
    "Green": "#6dff6d", "Red": "#ff5555", "Orange": "#ffa733",
    "Pink": "#ff7ae0", "Black": "#000000",
}
BOARDS = [("Dark board", "#101418"), ("Black board", "#000000"), ("White board", "#ffffff")]
WATERMARK = "Made by NikilPranav"

REC_STYLE = "background:#d32f2f;color:white;font-weight:bold;padding:8px 16px;"
STOP_STYLE = "background:#333;color:white;font-weight:bold;padding:8px 16px;"
CAM_RETRY_MS = 2000
CAM_MAX_RETRIES = 60

DEFAULTS = {
    "bg": "#101418",
    "pen": "#ffffff",
    "corner": CAM_CORNERS[0],
    "size": CAM_SIZES[1],
    "width": 4.0,
    "startup": True,
}


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ------------------------------------------------------------------- webcam
class Webcam:
    """Grabs webcam frames in a background thread (centre-cropped square, RGB)."""

    def __init__(self):
        self.frame = None
        self.running = False

    def start(self):
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, api)
        if not cap.isOpened():
            cap.release()
            return False
        self.running = True
        threading.Thread(target=self._loop, args=(cap,), daemon=True).start()
        return True

    def _loop(self, cap):
        while self.running:
            ok, img = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            h, w = img.shape[:2]
            m = min(h, w)
            y0, x0 = (h - m) // 2, (w - m) // 2
            sq = cv2.resize(img[y0:y0 + m, x0:x0 + m], (CAM_RES, CAM_RES),
                            interpolation=cv2.INTER_AREA)
            self.frame = np.ascontiguousarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
        cap.release()

    def stop(self):
        self.running = False
        self.frame = None


# -------------------------------------------------------------------- board
class Board(QWidget):
    def __init__(self):
        super().__init__()
        self.bg = QColor("#101418")
        self.image = QImage(CANVAS_W, CANVAS_H, QImage.Format.Format_RGB32)
        self.image.fill(self.bg)
        self.pen_color = QColor("#ffffff")
        self.base_width = 4.0
        self.eraser_on = False
        self.undo_stack = []
        self.drawing = False
        self.last = None
        self.laser_on = False
        self.laser_pos = None
        self.cam = Webcam()
        self.cam_on = False
        self.cam_diam = CAM_SIZES[1]
        self.cam_corner = CAM_CORNERS[0]
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)

    def _view(self):
        s = min(self.width() / CANVAS_W, self.height() / CANVAS_H)
        w, h = CANVAS_W * s, CANVAS_H * s
        return s, (self.width() - w) / 2, (self.height() - h) / 2

    def _to_canvas(self, p):
        s, x0, y0 = self._view()
        return QPointF((p.x() - x0) / s, (p.y() - y0) / s)

    def _draw_watermark(self, p):
        """Permanent badge at the bottom-centre. Visible on screen and in video."""
        font = p.font()
        font.setPointSize(20)
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(WATERMARK)
        th = fm.height()
        pad_x, pad_y = 18, 10
        box_w = tw + pad_x * 2
        box_h = th + pad_y
        x = (CANVAS_W - box_w) / 2
        y = CANVAS_H - box_h - 22
        rect = QRectF(x, y, box_w, box_h)

        if self.bg.lightness() < 128:
            fill = QColor(0, 0, 0, 120)
            fg = QColor(255, 255, 255, 210)
        else:
            fill = QColor(255, 255, 255, 210)
            fg = QColor(0, 0, 0, 190)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(rect, 14, 14)
        p.setPen(fg)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, WATERMARK)
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_overlays(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = self.cam.frame if self.cam_on else None
        if frame is not None:
            d, m = self.cam_diam, 30
            x = m if "left" in self.cam_corner else CANVAS_W - d - m
            y = m if "top" in self.cam_corner else CANVAS_H - d - m
            rect = QRectF(x, y, d, d)
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                          frame.strides[0], QImage.Format.Format_RGB888)
            path = QPainterPath()
            path.addEllipse(rect)
            p.save()
            p.setClipPath(path)
            p.drawImage(rect, qimg)
            p.restore()
            p.setPen(QPen(QColor("white"), 5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(rect)
        if self.laser_pos is not None:
            c = self.laser_pos
            g = QRadialGradient(c, 34)
            g.setColorAt(0.0, QColor(255, 60, 60, 255))
            g.setColorAt(0.3, QColor(255, 30, 30, 220))
            g.setColorAt(1.0, QColor(255, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(c, 34, 34)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(c, 4, 4)
        self._draw_watermark(p)

    def render_frame(self):
        img = self.image.copy()
        p = QPainter(img)
        self._draw_overlays(p)
        p.end()
        return img

    def paintEvent(self, _):
        s, x0, y0 = self._view()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#222"))
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(x0, y0)
        p.scale(s, s)
        p.drawImage(0, 0, self.image)
        self._draw_overlays(p)

    def push_undo(self):
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.image = self.undo_stack.pop()
            self.update()

    def clear(self):
        self.push_undo()
        self.image.fill(self.bg)
        self.update()

    def set_background(self, color, clear=True):
        self.bg = QColor(color)
        if clear:
            self.clear()
        else:
            self.update()

    def set_laser(self, on):
        self.laser_on = on
        self.laser_pos = None
        self.drawing = False
        self.update()

    def _segment(self, p0, p1, pressure, erase):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if erase:
            pen = QPen(self.bg, 45)
        else:
            width = self.base_width * (0.25 + 1.25 * max(0.0, min(1.0, pressure)))
            pen = QPen(self.pen_color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        painter.end()
        self.update()

    def _press(self, pos, pressure, erase):
        self.push_undo()
        self.drawing = True
        self.last = self._to_canvas(pos)
        self._segment(self.last, self.last, pressure, erase)

    def _move(self, pos, pressure, erase):
        if not self.drawing:
            return
        cur = self._to_canvas(pos)
        self._segment(self.last, cur, pressure, erase)
        self.last = cur

    def _release(self):
        self.drawing = False

    def tabletEvent(self, e):
        t = e.type()
        if t == QEvent.Type.TabletLeaveProximity:
            self.laser_pos = None
            self.update()
            e.accept()
            return
        if self.laser_on:
            self.laser_pos = self._to_canvas(e.position())
            self.update()
            e.accept()
            return
        erase = (self.eraser_on or
                 e.pointerType() == QPointingDevice.PointerType.Eraser)
        if t == QEvent.Type.TabletPress:
            self._press(e.position(), e.pressure(), erase)
        elif t == QEvent.Type.TabletMove:
            self._move(e.position(), e.pressure(), erase)
        elif t == QEvent.Type.TabletRelease:
            self._release()
        e.accept()

    def mousePressEvent(self, e):
        if self.laser_on:
            self.laser_pos = self._to_canvas(e.position())
            self.update()
        elif e.button() == Qt.MouseButton.LeftButton:
            self._press(e.position(), 0.6, self.eraser_on)

    def mouseMoveEvent(self, e):
        if self.laser_on:
            self.laser_pos = self._to_canvas(e.position())
            self.update()
        else:
            self._move(e.position(), 0.6, self.eraser_on)

    def mouseReleaseEvent(self, e):
        self._release()

    def leaveEvent(self, e):
        if self.laser_pos is not None:
            self.laser_pos = None
            self.update()


# ----------------------------------------------------------- settings dialog
class SettingsDialog(QDialog):
    """Startup tab where the teacher sets board look and webcam placement."""

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        cfg = dict(cfg or DEFAULTS)
        self.setWindowTitle("Physics Software - Settings")
        self.setMinimumWidth(420)

        form = QFormLayout()

        self.bg_combo = QComboBox()
        for label, val in BOARDS:
            self.bg_combo.addItem(label, val)
        idx = self.bg_combo.findData(cfg["bg"])
        self.bg_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.bg_combo.currentIndexChanged.connect(self._bg_changed)
        form.addRow("Board colour:", self.bg_combo)

        self.pen_combo = QComboBox()
        for name, hexcol in COLORS.items():
            self.pen_combo.addItem(name, hexcol)
        idx = self.pen_combo.findData(cfg["pen"])
        self.pen_combo.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("Default pen colour:", self.pen_combo)

        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(1, 20)
        self.width_slider.setValue(int(cfg["width"]))
        form.addRow("Default pen width:", self.width_slider)

        self.corner_combo = QComboBox()
        self.corner_combo.addItems(CAM_CORNERS)
        self.corner_combo.setCurrentText(cfg["corner"])
        form.addRow("Webcam position:", self.corner_combo)

        self.size_combo = QComboBox()
        for label, val in (("Small", CAM_SIZES[0]), ("Medium", CAM_SIZES[1]),
                           ("Large", CAM_SIZES[2])):
            self.size_combo.addItem(label, val)
        idx = self.size_combo.findData(cfg["size"])
        self.size_combo.setCurrentIndex(idx if idx >= 0 else 1)
        form.addRow("Webcam size:", self.size_combo)

        self.startup_check = QCheckBox("Show this settings tab when the app starts")
        self.startup_check.setChecked(bool(cfg["startup"]))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.startup_check)
        lay.addWidget(buttons)

    def _bg_changed(self):
        dark = QColor(self.bg_combo.currentData()).lightness() < 128
        want = "#ffffff" if dark else "#000000"
        idx = self.pen_combo.findData(want)
        if idx >= 0:
            self.pen_combo.setCurrentIndex(idx)

    def values(self):
        return {
            "bg": self.bg_combo.currentData(),
            "pen": self.pen_combo.currentData(),
            "corner": self.corner_combo.currentText(),
            "size": int(self.size_combo.currentData()),
            "width": float(self.width_slider.value()),
            "startup": self.startup_check.isChecked(),
        }


# --------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physics Software")
        self.resize(1280, 780)
        self.board = Board()
        self.setCentralWidget(self.board)

        self.store = QSettings("PhysicsSoftware", "PhysicsBoard")
        self.cfg = self._load_settings()

        self.recording = False
        self.paused = False
        self.paused_total = 0.0
        self.pause_started = 0.0
        self.writer = None
        self.t0 = 0.0
        self.frames_written = 0
        self.raw_video = ""
        self.base = ""
        self.audio_chunks = []
        self.stream = None
        self.audio_rate = AUDIO_RATE
        self.audio_error = ""
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._grab_frame)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick_label)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.board.update)
        self.finished.connect(self._on_saved)
        self._cam_retries = 0

        self._build_toolbars()
        self._apply_settings(self.cfg, clear_board=False)
        self._start_webcam()
        self.statusBar().showMessage("Ready. Press Settings to change board or camera options.")

        if self.cfg.get("startup", True):
            QTimer.singleShot(200, self._open_settings)

    def _load_settings(self):
        s = self.store
        return {
            "bg": s.value("bg", DEFAULTS["bg"]),
            "pen": s.value("pen", DEFAULTS["pen"]),
            "corner": s.value("corner", DEFAULTS["corner"]),
            "size": int(s.value("size", DEFAULTS["size"])),
            "width": float(s.value("width", DEFAULTS["width"])),
            "startup": s.value("startup", DEFAULTS["startup"]) in (True, "true", "True", "1"),
        }

    def _save_settings(self):
        s = self.store
        s.setValue("bg", self.cfg["bg"])
        s.setValue("pen", self.cfg["pen"])
        s.setValue("corner", self.cfg["corner"])
        s.setValue("size", self.cfg["size"])
        s.setValue("width", self.cfg["width"])
        s.setValue("startup", self.cfg["startup"])
        s.sync()

    def _apply_settings(self, cfg, clear_board=True):
        self.board.set_background(cfg["bg"], clear=clear_board)
        self.board.pen_color = QColor(cfg["pen"])
        self.board.base_width = float(cfg["width"])
        if cfg["corner"] in CAM_CORNERS:
            self.board.cam_corner = cfg["corner"]
        if cfg["size"] in CAM_SIZES:
            self.board.cam_diam = cfg["size"]
        if hasattr(self, "corner_btn"):
            self.corner_btn.setText(f"Corner: {self.board.cam_corner}")
        if hasattr(self, "camsize_btn"):
            self.camsize_btn.setText("Cam size: " + "SML"[CAM_SIZES.index(self.board.cam_diam)])
        if hasattr(self, "width_slider"):
            self.width_slider.blockSignals(True)
            self.width_slider.setValue(int(self.board.base_width))
            self.width_slider.blockSignals(False)
        self.board.update()

    def _open_settings(self):
        dlg = SettingsDialog(self, self.cfg)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.cfg = dlg.values()
            self._save_settings()
            self._apply_settings(self.cfg, clear_board=False)
            self.statusBar().showMessage("Settings saved.")

    def _build_toolbars(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        self.rec_btn = QPushButton("Start Recording")
        self.rec_btn.setStyleSheet(REC_STYLE)
        self.rec_btn.clicked.connect(self.toggle_recording)
        tb.addWidget(self.rec_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("padding:8px 14px;")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        tb.addWidget(self.pause_btn)

        self.timer_label = QLabel("  00:00  ")
        self.timer_label.setStyleSheet("font-size:16px;font-weight:bold;")
        tb.addWidget(self.timer_label)
        tb.addSeparator()

        self.laser_btn = QPushButton("Laser Pointer (L)")
        self.laser_btn.setCheckable(True)
        self.laser_btn.toggled.connect(self.board.set_laser)
        tb.addWidget(self.laser_btn)
        tb.addSeparator()

        self.cam_status = QLabel("  Camera: starting…  ")
        self.cam_status.setStyleSheet("font-weight:bold;padding:4px 10px;")
        tb.addWidget(self.cam_status)

        self.corner_btn = QPushButton("Corner: bottom-right")
        self.corner_btn.clicked.connect(self._cycle_corner)
        tb.addWidget(self.corner_btn)

        self.camsize_btn = QPushButton("Cam size: M")
        self.camsize_btn.clicked.connect(self._cycle_cam_size)
        tb.addWidget(self.camsize_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self._open_settings)
        tb.addWidget(self.settings_btn)

        self.addToolBarBreak()

        tb2 = QToolBar()
        tb2.setMovable(False)
        self.addToolBar(tb2)

        for name, hexcol in COLORS.items():
            b = QPushButton()
            b.setFixedSize(28, 28)
            b.setToolTip(name)
            b.setStyleSheet(f"background:{hexcol};border:2px solid #888;border-radius:14px;")
            b.clicked.connect(lambda _, c=hexcol: self._set_color(c))
            tb2.addWidget(b)
        tb2.addSeparator()

        tb2.addWidget(QLabel(" Size "))
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(1, 20)
        self.width_slider.setValue(4)
        self.width_slider.setFixedWidth(110)
        self.width_slider.valueChanged.connect(lambda v: setattr(self.board, "base_width", float(v)))
        tb2.addWidget(self.width_slider)
        tb2.addSeparator()

        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.toggled.connect(lambda on: setattr(self.board, "eraser_on", on))
        tb2.addWidget(self.eraser_btn)

        for text, fn in (("Undo", self.board.undo),
                         ("Clear", self.board.clear),
                         ("Dark/White Board", self._flip_bg),
                         ("Save Image", self._save_png),
                         ("Fullscreen", self._toggle_fullscreen)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            tb2.addWidget(b)

        for key, fn in (("Ctrl+Z", self.board.undo), ("Ctrl+R", self.toggle_recording),
                        ("Ctrl+P", self.toggle_pause), ("L", self.laser_btn.toggle),
                        ("Ctrl+,", self._open_settings), ("F11", self._toggle_fullscreen)):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            self.addAction(a)

    def _set_color(self, c):
        self.board.pen_color = QColor(c)
        self.eraser_btn.setChecked(False)
        self.laser_btn.setChecked(False)

    def _flip_bg(self):
        dark = self.board.bg.lightness() < 128
        self.board.set_background("#ffffff" if dark else "#101418")
        self.board.pen_color = QColor("#000000" if dark else "#ffffff")
        self.cfg["bg"] = self.board.bg.name()
        self.cfg["pen"] = self.board.pen_color.name()
        self._save_settings()

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _save_png(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        default = os.path.join(OUT_DIR, datetime.now().strftime("board_%Y%m%d_%H%M%S.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Save board", default, "PNG (*.png)")
        if path:
            self.board.image.save(path)

    def _start_webcam(self):
        if self.board.cam.start():
            self.board.cam_on = True
            self.ui_timer.start(33)
            self.cam_status.setText("  Camera: On  ")
            self.cam_status.setStyleSheet("font-weight:bold;color:#2e7d32;padding:4px 10px;")
            self._cam_retries = 0
        else:
            self.cam_status.setText("  Camera: waiting…  ")
            self.cam_status.setStyleSheet("font-weight:bold;color:#c62828;padding:4px 10px;")
            if self._cam_retries < CAM_MAX_RETRIES:
                self._cam_retries += 1
                QTimer.singleShot(CAM_RETRY_MS, self._start_webcam)
            else:
                self.cam_status.setText("  Camera: not found  ")
                self.statusBar().showMessage(
                    "Webcam not detected. Close other apps using it, check the USB cable, then restart."
                )

    def _cycle_corner(self):
        i = (CAM_CORNERS.index(self.board.cam_corner) + 1) % len(CAM_CORNERS)
        self.board.cam_corner = CAM_CORNERS[i]
        self.corner_btn.setText(f"Corner: {CAM_CORNERS[i]}")
        self.cfg["corner"] = CAM_CORNERS[i]
        self._save_settings()
        self.board.update()

    def _cycle_cam_size(self):
        i = (CAM_SIZES.index(self.board.cam_diam) + 1) % len(CAM_SIZES)
        self.board.cam_diam = CAM_SIZES[i]
        self.camsize_btn.setText("Cam size: " + "SML"[i])
        self.cfg["size"] = CAM_SIZES[i]
        self._save_settings()
        self.board.update()

    def toggle_recording(self):
        self.stop_recording() if self.recording else self.start_recording()

    def start_recording(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base = os.path.join(OUT_DIR, f"lecture_{stamp}")
        self.raw_video = self.base + "_raw.mp4"

        self.writer = cv2.VideoWriter(
            self.raw_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (CANVAS_W, CANVAS_H)
        )

        if not self.writer.isOpened():
            self.statusBar().showMessage("Could not start the video writer.")
            return

        self.audio_chunks = []
        self.stream = None
        self.audio_rate = AUDIO_RATE
        self.audio_error = ""
        self.frames_written = 0
        self.paused = False
        self.paused_total = 0.0
        self.t0 = time.perf_counter()
        self.recording = True

        if sd is None:
            self.audio_error = "sounddevice is not installed."
        else:
            try:
                device_id = sd.default.device[0]

                if device_id is None or device_id < 0:
                    raise RuntimeError("No default microphone is selected.")

                device = sd.query_devices(device_id, "input")

                if int(device["max_input_channels"]) < 1:
                    raise RuntimeError("Selected device has no microphone input.")

                preferred_rate = int(
                    device.get("default_samplerate") or AUDIO_RATE
                )

                rates = list(dict.fromkeys([
                    preferred_rate, AUDIO_RATE, 48000
                ]))

                last_error = None

                for rate in rates:
                    try:
                        sd.check_input_settings(
                            device=device_id,
                            channels=1,
                            dtype="float32",
                            samplerate=rate
                        )

                        stream = sd.InputStream(
                            device=device_id,
                            samplerate=rate,
                            channels=1,
                            dtype="float32",
                            callback=self._audio_cb
                        )

                        stream.start()
                        self.stream = stream
                        self.audio_rate = rate
                        break

                    except Exception as ex:
                        last_error = ex

                if self.stream is None:
                    raise RuntimeError(
                        str(last_error or "Could not open microphone.")
                    )

            except Exception as ex:
                self.stream = None
                self.audio_error = str(ex)

        if self.stream is None:
            self.statusBar().showMessage(
                f"Audio unavailable: {self.audio_error} Recording video only."
            )
        else:
            self.statusBar().showMessage(
                f"Recording video and microphone audio ({self.audio_rate} Hz)."
            )

        self.rec_timer.start(int(1000 / FPS / 2))
        self.clock.start(500)
        self.rec_btn.setText("Stop Recording")
        self.rec_btn.setStyleSheet(STOP_STYLE)
        self.pause_btn.setEnabled(True)

    def _audio_cb(self, indata, frames, time_info, status):
        if self.recording and not self.paused:
            self.audio_chunks.append(indata.copy())

    def _active_elapsed(self):
        now = time.perf_counter()
        cur_pause = (now - self.pause_started) if self.paused else 0.0
        return now - self.t0 - self.paused_total - cur_pause

    def toggle_pause(self):
        if not self.recording:
            return

        if not self.paused:
            self.paused = True
            self.pause_started = time.perf_counter()
            self.pause_btn.setText("Resume")
            self.statusBar().showMessage(
                "Recording paused - nothing is being captured."
            )
        else:
            self.paused_total += time.perf_counter() - self.pause_started
            self.paused = False
            self.pause_btn.setText("Pause")
            self.statusBar().showMessage("Recording resumed.")

    def _grab_frame(self):
        if self.paused or not self.recording:
            return

        due = int(self._active_elapsed() * FPS) - self.frames_written

        if due <= 0:
            return

        img = self.board.render_frame().convertToFormat(
            QImage.Format.Format_RGB888
        )

        raw = img.constBits().asstring(img.sizeInBytes())
        arr = np.frombuffer(raw, np.uint8).reshape(
            img.height(), img.bytesPerLine()
        )

        frame = cv2.cvtColor(
            arr[:, :img.width() * 3].reshape(
                img.height(), img.width(), 3
            ),
            cv2.COLOR_RGB2BGR
        )

        for _ in range(min(due, FPS)):
            self.writer.write(frame)
            self.frames_written += 1

    def _tick_label(self):
        s = int(self._active_elapsed())

        if self.paused:
            self.timer_label.setText(
                f"  PAUSED {s // 60:02d}:{s % 60:02d}  "
            )
        else:
            self.timer_label.setText(
                f"  {s // 60:02d}:{s % 60:02d}  "
            )

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        self.rec_timer.stop()
        self.clock.stop()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        wav = None

        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass

            try:
                self.stream.close()
            except Exception:
                pass

            self.stream = None

            if self.audio_chunks:
                try:
                    data = np.concatenate(self.audio_chunks, axis=0)
                    data = np.clip(data, -1.0, 1.0)
                    data = (data * 32767).astype(np.int16)

                    wav = self.base + ".wav"

                    with wave.open(wav, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(self.audio_rate)
                        w.writeframes(data.tobytes())

                except Exception as ex:
                    wav = None
                    self.statusBar().showMessage(
                        f"Audio export failed: {ex}"
                    )

        self.audio_chunks = []
        self.paused = False
        self.rec_btn.setText("Start Recording")
        self.rec_btn.setStyleSheet(REC_STYLE)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.timer_label.setText("  00:00  ")
        self.statusBar().showMessage("Saving video, please wait...")

        threading.Thread(
            target=self._finalize,
            args=(wav,),
            daemon=True
        ).start()

    def _finalize(self, wav):
        final = self.base + ".mp4"
        ffmpeg = find_ffmpeg()
        try:
            if ffmpeg:
                cmd = [ffmpeg, "-y", "-i", self.raw_video]
                if wav:
                    cmd += ["-i", wav]
                cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "20"]
                if wav:
                    cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
                cmd += [final]
                subprocess.run(cmd, check=True, capture_output=True)
                os.remove(self.raw_video)
                if wav:
                    os.remove(wav)
            else:
                os.replace(self.raw_video, final)
        except Exception as ex:
            self.finished.emit(f"Saved raw files in {OUT_DIR} (conversion failed: {ex})")
            return
        self.finished.emit(f"Saved: {final}")

    def _on_saved(self, msg):
        self.statusBar().showMessage(msg)

    def closeEvent(self, e):
        if self.recording:
            self.stop_recording()
        self.board.cam.stop()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle("Physics Software - Made by NikilPranav")
    win.show()
    sys.exit(app.exec())
