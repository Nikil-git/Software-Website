"""
Physics Software - whiteboard + lecture recorder (Python 3, PyQt6)

Tuned for a low-power laptop (Dell Vostro 15-3568, Core i3-6006U, HD 520,
4-8 GB RAM, HDD):
  * 1280x720 @ 24 fps, streamed straight into ffmpeg (no huge raw file)
  * audio streamed to disk in chunks (no growing RAM list)
  * undo snapshots stored PNG-compressed in RAM (a flat whiteboard image
    compresses ~10-20x, so 10 undo steps cost a few MB instead of ~35 MB)
  * only 3 worker threads (webcam, video writer, audio writer); ffmpeg
    limited to 2 threads, below-normal priority; coarse Qt timers to cut
    wake-ups; webcam at 640x480/15fps internally, shown at 200px
  * unchanged frames are reused rather than re-rendered; the screen
    repaints only the area that changed
  * Intel Quick Sync (h264_qsv) used automatically if ffmpeg supports it

License gate (see LICENSE section below):
  * on startup, and every few hours while running, the app checks a small
    JSON status file you host (e.g. a GitHub Gist raw URL)
  * {"status": "active"}  -> works normally
  * {"status": "locked", "message": "..."}  -> the board and the Start
    Recording button are disabled and your message is shown; nothing he
    already recorded is touched or deleted
  * if there is no internet, the last known status is kept for a grace
    period (default 3 days) before the app locks itself as a precaution

pip install PyQt6 opencv-python numpy sounddevice imageio-ffmpeg
(For Quick Sync, put a full ffmpeg build on PATH, e.g. from gyan.dev.)
"""
import os
import sys
import time
import wave
import queue
import shutil
import threading
import subprocess
import urllib.request
import json
from datetime import datetime

import numpy as np
import cv2
from PyQt6.QtCore import (Qt, QTimer, QPointF, QRectF, QEvent, pyqtSignal,
                          QSettings, QBuffer, QIODevice, QByteArray)
from PyQt6.QtGui import (QImage, QPainter, QPen, QColor, QAction, QKeySequence,
                         QPointingDevice, QPainterPath, QRadialGradient, QBrush,
                         QFont, QFontMetrics)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QToolBar,
                             QPushButton, QLabel, QSlider, QFileDialog,
                             QDialog, QFormLayout, QComboBox, QCheckBox,
                             QDialogButtonBox, QVBoxLayout)

try:
    import sounddevice as sd
except Exception:
    sd = None

# ----------------------------------------------------------------- settings
CANVAS_W, CANVAS_H = 1280, 720        # video size (set 1920,1080 only on a faster PC)
S = CANVAS_H / 1080.0
FPS = 24
AUDIO_RATE = 44100
OUT_DIR = os.path.join(os.path.expanduser("~"), "Videos", "PhysicsBoard")
MAX_UNDO = 12                          # cheap now: snapshots are PNG-compressed
VIDEO_QUEUE = 6
IO_BUFFER = 1 << 20
CAM_RES = 200                           # smaller = less CPU (tune down further if needed)
CAM_FPS = 15
CAM_RETRY_S = 2.0
CAM_SIZES = [int(200 * S), int(280 * S), int(380 * S)]
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

DEFAULTS = {
    "bg": "#101418", "pen": "#ffffff", "corner": CAM_CORNERS[0],
    "size": CAM_SIZES[1], "width": 4.0, "startup": True,
}

WIN = sys.platform == "win32"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if WIN else 0
BELOW_NORMAL = 0x00004000 if WIN else 0

# --------------------------------------------------------------- LICENSE ---
# Host a tiny JSON file somewhere you control (a GitHub Gist's "raw" URL is
# the easiest: create a gist with a file named status.json, then paste its
# raw link below). Edit that file whenever you want to lock or unlock.
#   Unlocked : {"status": "active"}
#   Locked   : {"status": "locked", "message": "Please contact Nikil to renew."}
LICENSE_URL = "https://gist.githubusercontent.com/YOUR_USERNAME/YOUR_GIST_ID/raw/status.json"
LICENSE_CHECK_INTERVAL_S = 6 * 3600     # re-check every 6 hours while running
LICENSE_GRACE_S = 3 * 24 * 3600         # work offline up to 3 days on the last known state
LICENSE_TIMEOUT_S = 8


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_encoder(ffmpeg):
    """Returns 'h264_qsv' (Intel Quick Sync) if it really works, else 'libx264'."""
    if not ffmpeg:
        return "libx264"
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "testsrc=size=640x360:rate=10:duration=0.5", "-c:v", "h264_qsv",
             "-f", "null", "-"],
            capture_output=True, timeout=20, creationflags=NO_WINDOW)
        return "h264_qsv" if r.returncode == 0 else "libx264"
    except Exception:
        return "libx264"


# ------------------------------------------------------------------- license
class LicenseGate:
    """Background-checks LICENSE_URL and reports ("active"/"locked"/"unknown", message).
    Never touches recorded files; it only tells MainWindow whether to allow use.
    Fails safe-ish: no internet keeps the last known state for LICENSE_GRACE_S,
    then locks with a "couldn't verify" message so the app can't be used forever
    offline, but a short outage never interrupts a lecture."""

    def __init__(self, settings, on_status, url=None):
        self.settings = settings
        self.on_status = on_status      # called from a worker thread with (status, message)
        self.url = url if url is not None else LICENSE_URL   # read fresh, not frozen at import
        self._stop = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop = True

    def check_once(self):
        """One check; updates the cache and calls on_status. Safe to call off the UI thread."""
        status, message = self._fetch()
        now = time.time()
        if status in ("active", "locked"):
            self.settings.setValue("lic/status", status)
            self.settings.setValue("lic/message", message)
            self.settings.setValue("lic/ts", now)
            self.settings.sync()
            self.on_status(status, message)
            return
        # fetch failed: fall back to the cached state within the grace period
        cached = self.settings.value("lic/status", "")
        ts = float(self.settings.value("lic/ts", 0) or 0)
        if cached in ("active", "locked") and (now - ts) < LICENSE_GRACE_S:
            self.on_status(cached, self.settings.value("lic/message", ""))
        elif cached == "":
            # very first run with no internet at all: don't brick a fresh install
            self.on_status("active", "")
        else:
            self.on_status("locked", "Could not verify the licence (no internet for "
                                     "too long). Please connect to the internet.")

    def _fetch(self):
        try:
            req = urllib.request.Request(self.url, headers={"Cache-Control": "no-cache"})
            with urllib.request.urlopen(req, timeout=LICENSE_TIMEOUT_S) as r:
                data = json.loads(r.read().decode("utf-8"))
            status = str(data.get("status", "")).lower()
            if status not in ("active", "locked"):
                return "unknown", ""
            return status, str(data.get("message", "Please contact the developer to renew."))
        except Exception:
            return "unknown", ""

    def _loop(self):
        self.check_once()
        while not self._stop:
            for _ in range(int(LICENSE_CHECK_INTERVAL_S * 10)):
                if self._stop:
                    return
                time.sleep(0.1)
            self.check_once()


# ------------------------------------------------------------------- webcam
class Webcam:
    """Background thread: opens the webcam, keeps the latest square RGB frame,
    reconnects by itself if the camera is missing or unplugged.
    state: "off" | "waiting" | "on".  seq increases whenever frame changes."""

    def __init__(self):
        self.frame = None
        self.seq = 0
        self.running = False
        self.state = "off"
        self._gen = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self._gen += 1
        self.state = "waiting"
        threading.Thread(target=self._run, args=(self._gen,), daemon=True).start()

    def stop(self):
        self.running = False
        self.frame = None
        self.seq += 1
        self.state = "off"

    def _alive(self, gen):
        return self.running and gen == self._gen

    def _wait(self, gen, seconds):
        end = time.time() + seconds
        while self._alive(gen) and time.time() < end:
            time.sleep(0.1)

    def _open(self):
        api = cv2.CAP_DSHOW if WIN else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, api)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self, gen):
        interval = 1.0 / CAM_FPS
        while self._alive(gen):
            cap = self._open()
            if cap is None:
                if gen == self._gen:
                    self.state = "waiting"
                self._wait(gen, CAM_RETRY_S)
                continue
            self.state = "on"
            fails, last = 0, 0.0
            while self._alive(gen):
                if not cap.grab():
                    fails += 1
                    if fails >= 40:
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                now = time.time()
                if now - last < interval:
                    continue
                ok, img = cap.retrieve()
                if not ok or img is None:
                    continue
                last = now
                h, w = img.shape[:2]
                m = min(h, w)
                y0, x0 = (h - m) // 2, (w - m) // 2
                sq = cv2.resize(img[y0:y0 + m, x0:x0 + m], (CAM_RES, CAM_RES),
                                interpolation=cv2.INTER_AREA)
                self.frame = np.ascontiguousarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
                self.seq += 1
            cap.release()
            if gen == self._gen:
                self.frame = None
                self.seq += 1
                self.state = "waiting" if self.running else "off"


# ----------------------------------------------------------------- recorder
class Recorder:
    """Streams frames to ffmpeg and microphone audio to a wav file while
    recording, then muxes (no re-encoding) when stopped."""

    def __init__(self, base, ffmpeg, encoder, on_done):
        self.base = base
        self.ffmpeg = ffmpeg
        self.encoder = encoder
        self.on_done = on_done
        self.video_tmp = base + "_video.mp4"
        self.wav = base + ".wav"
        self.final = base + ".mp4"
        self.log_path = base + "_ffmpeg.log"
        self.vq = queue.Queue(maxsize=VIDEO_QUEUE)
        self.aq = None
        self.proc = None
        self.cvw = None
        self.log = None
        self.stream = None
        self.wf = None
        self.wfile = None
        self.athread = None
        self.vthread = None
        self.rate = AUDIO_RATE
        self.paused = False
        self.frames = 0
        self.error = ""
        self.audio_error = ""

    def start(self, use_mic=True):
        os.makedirs(OUT_DIR, exist_ok=True)
        if self.ffmpeg:
            cmd = [self.ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
                   "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(FPS), "-i", "-", "-an"]
            if self.encoder == "h264_qsv":
                cmd += ["-c:v", "h264_qsv", "-preset", "veryfast",
                        "-global_quality", "27", "-look_ahead", "0", "-pix_fmt", "nv12"]
            else:
                cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
                        "-threads", "2", "-pix_fmt", "yuv420p"]
            cmd += ["-g", str(FPS * 2), self.video_tmp]
            self.log = open(self.log_path, "wb")
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=NO_WINDOW | BELOW_NORMAL)
        else:
            self.cvw = cv2.VideoWriter(self.video_tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                                       FPS, (CANVAS_W, CANVAS_H))
            if not self.cvw.isOpened():
                raise RuntimeError("Could not start the video writer.")
        if use_mic:
            self._open_mic()
        self.vthread = threading.Thread(target=self._video_loop, daemon=True)
        self.vthread.start()

    def _open_mic(self):
        if sd is None:
            self.audio_error = "sounddevice is not installed."
            return
        try:
            dev = sd.query_devices(kind="input")
            dev_id = sd.default.device[0]
            if dev_id is None or dev_id < 0:
                dev_id = None
            if int(dev.get("max_input_channels", 0)) < 1:
                raise RuntimeError("No microphone input device is available.")
            rates = []
            for r in (int(dev.get("default_samplerate") or AUDIO_RATE),
                      AUDIO_RATE, 48000, 44100, 32000, 22050):
                if r not in rates:
                    rates.append(r)
            last = None
            for r in rates:
                try:
                    sd.check_input_settings(device=dev_id, channels=1,
                                            dtype="int16", samplerate=r)
                    self._begin_audio(r)
                    self.stream = sd.InputStream(device=dev_id, samplerate=r, channels=1,
                                                 dtype="int16", latency="high",
                                                 callback=self._audio_cb)
                    self.stream.start()
                    return
                except Exception as ex:
                    last = ex
                    self._abort_audio()
            raise RuntimeError(str(last or "Could not open the microphone."))
        except Exception as ex:
            self.audio_error = str(ex)

    def _begin_audio(self, rate):
        self.rate = rate
        self.aq = queue.Queue()
        self.wfile = open(self.wav, "wb", buffering=IO_BUFFER)
        self.wf = wave.open(self.wfile, "wb")
        self.wf.setnchannels(1)
        self.wf.setsampwidth(2)
        self.wf.setframerate(rate)
        self.athread = threading.Thread(target=self._audio_loop, daemon=True)
        self.athread.start()

    def _abort_audio(self):
        if self.aq is not None:
            self.aq.put(None)
            self.athread.join()
            self.aq = None
            try:
                os.remove(self.wav)
            except OSError:
                pass

    def _audio_cb(self, indata, frames, time_info, status):
        if not self.paused:
            self.aq.put_nowait(indata.copy())

    def _audio_loop(self):
        try:
            while True:
                chunk = self.aq.get()
                if chunk is None:
                    break
                self.wf.writeframes(chunk.tobytes())
        finally:
            self.wf.close()
            self.wfile.close()

    def push(self, data, count):
        for _ in range(count):
            try:
                self.vq.put_nowait(data)
            except queue.Full:
                break
            self.frames += 1

    def _write(self, data):
        if self.proc is not None:
            self.proc.stdin.write(data)
        else:
            arr = np.frombuffer(data, np.uint8).reshape(CANVAS_H, CANVAS_W, 4)
            self.cvw.write(np.ascontiguousarray(arr[:, :, :3]))

    def _video_loop(self):
        ok = True
        while True:
            data = self.vq.get()
            if data is None:
                break
            if ok:
                try:
                    self._write(data)
                except Exception as ex:
                    ok = False
                    self.error = str(ex)
        self._finish(ok)

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.vq.put(None)

    def _finish(self, ok):
        hw_failed = False
        try:
            rc = 0
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
                rc = self.proc.wait()
                self.log.close()
            elif self.cvw is not None:
                self.cvw.release()
            if self.aq is not None:
                self.aq.put(None)
                self.athread.join()
            if rc != 0 or not ok:
                hw_failed = self.encoder != "libx264"
                extra = " Quick Sync failed - the software encoder will be used next time." \
                    if hw_failed else ""
                self.on_done(f"Encoding failed (see {self.log_path}).{extra}", hw_failed)
                return
            has_audio = (self.aq is not None and os.path.exists(self.wav)
                         and os.path.getsize(self.wav) > 44)
            note = ""
            if has_audio and self.ffmpeg:
                subprocess.run(
                    [self.ffmpeg, "-y", "-i", self.video_tmp, "-i", self.wav,
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", self.final],
                    check=True, capture_output=True, creationflags=NO_WINDOW)
                os.remove(self.video_tmp)
                os.remove(self.wav)
            else:
                os.replace(self.video_tmp, self.final)
                if has_audio:
                    note = " (ffmpeg not found: audio kept as a separate .wav file)"
            try:
                os.remove(self.log_path)
            except OSError:
                pass
            self.on_done(f"Saved: {self.final}{note}", False)
        except Exception as ex:
            self.on_done(f"Saved raw files in {OUT_DIR} (finishing failed: {ex})", False)


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
        self.undo_stack = []            # list of (QByteArray PNG, background colour)
        self.drawing = False
        self.last = None
        self.laser_on = False
        self.laser_pos = None
        self.laser_r = int(34 * S)
        self.cam = Webcam()
        self.cam_on = False
        self.cam_diam = CAM_SIZES[1]
        self.cam_corner = CAM_CORNERS[0]
        self.version = 0
        self.locked = False             # set by the license gate

        self._wm_font = QFont()
        self._wm_font.setPixelSize(int(26 * S))
        self._wm_font.setBold(True)
        fm = QFontMetrics(self._wm_font)
        bw = fm.horizontalAdvance(WATERMARK) + int(36 * S)
        bh = fm.height() + int(10 * S)
        self._wm_rect = QRectF((CANVAS_W - bw) / 2, CANVAS_H - bh - int(22 * S), bw, bh)

        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)

    def _view(self):
        s = min(self.width() / CANVAS_W, self.height() / CANVAS_H)
        return s, (self.width() - CANVAS_W * s) / 2, (self.height() - CANVAS_H * s) / 2

    def _to_canvas(self, p):
        s, x0, y0 = self._view()
        return QPointF((p.x() - x0) / s, (p.y() - y0) / s)

    def _to_widget_rect(self, r):
        s, x0, y0 = self._view()
        return QRectF(x0 + r.x() * s, y0 + r.y() * s,
                      r.width() * s, r.height() * s).toAlignedRect().adjusted(-2, -2, 2, 2)

    def _cam_rect(self):
        d, m = self.cam_diam, int(30 * S)
        x = m if "left" in self.cam_corner else CANVAS_W - d - m
        y = m if "top" in self.cam_corner else CANVAS_H - d - m
        return QRectF(x, y, d, d)

    def _laser_rect(self, pt):
        if pt is None:
            return None
        r = self.laser_r + 3
        return QRectF(pt.x() - r, pt.y() - r, 2 * r, 2 * r)

    def touch(self, *rects):
        self.version += 1
        if not rects:
            self.update()
            return
        for r in rects:
            if r is not None:
                self.update(self._to_widget_rect(r))

    def _draw_overlays(self, p):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = self.cam.frame if self.cam_on else None
        if frame is not None:
            rect = self._cam_rect()
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                          frame.strides[0], QImage.Format.Format_RGB888)
            path = QPainterPath()
            path.addEllipse(rect)
            p.save()
            p.setClipPath(path)
            p.drawImage(rect, qimg)
            p.restore()
            p.setPen(QPen(QColor("white"), 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(rect)
        if self.laser_pos is not None:
            c, r = self.laser_pos, self.laser_r
            g = QRadialGradient(c, r)
            g.setColorAt(0.0, QColor(255, 60, 60, 255))
            g.setColorAt(0.3, QColor(255, 30, 30, 220))
            g.setColorAt(1.0, QColor(255, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(g))
            p.drawEllipse(c, r, r)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(c, 3, 3)
        if self.bg.lightness() < 128:
            fill, fg = QColor(0, 0, 0, 120), QColor(255, 255, 255, 210)
        else:
            fill, fg = QColor(255, 255, 255, 210), QColor(0, 0, 0, 190)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(self._wm_rect, 12, 12)
        p.setFont(self._wm_font)
        p.setPen(fg)
        p.drawText(self._wm_rect, Qt.AlignmentFlag.AlignCenter, WATERMARK)

    def render_frame(self):
        img = self.image.copy()
        p = QPainter(img)
        self._draw_overlays(p)
        p.end()
        return img

    def paintEvent(self, ev):
        s, x0, y0 = self._view()
        p = QPainter(self)
        p.fillRect(ev.rect(), QColor("#222"))
        if abs(s - 1.0) > 0.01:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(x0, y0)
        p.scale(s, s)
        p.drawImage(0, 0, self.image)
        self._draw_overlays(p)

    # ---- undo: PNG-compressed in RAM (a flat whiteboard compresses hugely)
    def push_undo(self):
        buf = QByteArray()
        qbuf = QBuffer(buf)
        qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
        self.image.save(qbuf, "PNG")
        qbuf.close()
        self.undo_stack.append((buf, QColor(self.bg)))
        if len(self.undo_stack) > MAX_UNDO:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        buf, bg = self.undo_stack.pop()
        self.image.loadFromData(buf, "PNG")
        self.bg = bg
        self.touch()

    def clear(self):
        self.push_undo()
        self.image.fill(self.bg)
        self.touch()

    def set_background(self, color, undoable=True):
        if undoable:
            self.push_undo()
        self.bg = QColor(color)
        self.image.fill(self.bg)
        self.touch()

    def set_cam(self, corner=None, diam=None):
        old = self._cam_rect()
        if corner:
            self.cam_corner = corner
        if diam:
            self.cam_diam = diam
        self.touch(old, self._cam_rect())

    def set_laser(self, on):
        old = self.laser_pos
        self.laser_on = on
        self.laser_pos = None
        self.drawing = False
        self.touch(self._laser_rect(old))

    def _move_laser(self, pt):
        old = self.laser_pos
        self.laser_pos = pt
        self.touch(self._laser_rect(old), self._laser_rect(pt))

    def _segment(self, p0, p1, pressure, erase):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if erase:
            w = 45 * S
            pen = QPen(self.bg, w)
        else:
            w = self.base_width * (0.25 + 1.25 * max(0.0, min(1.0, pressure)))
            pen = QPen(self.pen_color, w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        painter.end()
        m = w + 4
        self.touch(QRectF(p0, p1).normalized().adjusted(-m, -m, m, m))

    def _press(self, pos, pressure, erase):
        if self.locked:
            return
        self.push_undo()
        self.drawing = True
        self.last = self._to_canvas(pos)
        self._segment(self.last, self.last, pressure, erase)

    def _move(self, pos, pressure, erase):
        if self.locked or not self.drawing:
            return
        cur = self._to_canvas(pos)
        self._segment(self.last, cur, pressure, erase)
        self.last = cur

    def _release(self):
        self.drawing = False

    def tabletEvent(self, e):
        t = e.type()
        if t == QEvent.Type.TabletLeaveProximity:
            self._move_laser(None)
            e.accept()
            return
        if self.locked:
            e.accept()
            return
        if self.laser_on:
            self._move_laser(self._to_canvas(e.position()))
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
        if self.locked:
            return
        if self.laser_on:
            self._move_laser(self._to_canvas(e.position()))
        elif e.button() == Qt.MouseButton.LeftButton:
            self._press(e.position(), 0.6, self.eraser_on)

    def mouseMoveEvent(self, e):
        if self.locked:
            return
        if self.laser_on:
            self._move_laser(self._to_canvas(e.position()))
        else:
            self._move(e.position(), 0.6, self.eraser_on)

    def mouseReleaseEvent(self, e):
        self._release()

    def leaveEvent(self, e):
        if self.laser_pos is not None:
            self._move_laser(None)


# ------------------------------------------------------------ lock overlay
class LockOverlay(QWidget):
    """Full-board translucent cover shown when the licence check says 'locked'.
    Purely visual/input-blocking - it never deletes or touches saved videos."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background: rgba(10,10,10,235);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        self.title = QLabel("Software locked")
        self.title.setStyleSheet("color:#ff6b6b;font-size:26px;font-weight:bold;")
        self.msg = QLabel("")
        self.msg.setStyleSheet("color:white;font-size:16px;")
        self.msg.setWordWrap(True)
        for w in (self.title, self.msg):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addStretch()
        lay.addWidget(self.title)
        lay.addWidget(self.msg)
        lay.addStretch()
        self.hide()

    def show_message(self, text):
        self.msg.setText(text or "Please contact the developer to continue using this software.")
        self.setGeometry(self.parent().rect())
        self.show()
        self.raise_()


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
        for label, val in zip(("Small", "Medium", "Large"), CAM_SIZES):
            self.size_combo.addItem(label, val)
        idx = self.size_combo.findData(cfg["size"])
        self.size_combo.setCurrentIndex(idx if idx >= 0 else 1)
        form.addRow("Webcam size:", self.size_combo)

        self.startup_check = QCheckBox("Show this settings tab when the app starts")
        self.startup_check.setChecked(bool(cfg["startup"]))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.startup_check)
        lay.addWidget(buttons)

    def _bg_changed(self):
        dark = QColor(self.bg_combo.currentData()).lightness() < 128
        idx = self.pen_combo.findData("#ffffff" if dark else "#000000")
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
    finished = pyqtSignal(str, bool)
    encoder_probed = pyqtSignal(str)
    license_status = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physics Software")
        self.resize(1280, 780)
        self.board = Board()
        self.setCentralWidget(self.board)
        self.lock_overlay = LockOverlay(self.board)

        self.store = QSettings("PhysicsSoftware", "PhysicsBoard")
        self.cfg = self._load_settings()

        self.rec = None
        self.paused = False
        self.paused_total = 0.0
        self.pause_started = 0.0
        self.t0 = 0.0
        self.workers = []
        self._frame_sig = None
        self._frame_bytes = None
        self._seen_cam_seq = -1

        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._grab_frame)
        self.clock = QTimer(self)
        self.clock.setTimerType(Qt.TimerType.CoarseTimer)   # fewer wake-ups, saves power
        self.clock.timeout.connect(self._tick_label)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._refresh_cam)
        self.cam_watch = QTimer(self)
        self.cam_watch.setTimerType(Qt.TimerType.CoarseTimer)
        self.cam_watch.timeout.connect(self._update_cam_label)
        self.finished.connect(self._on_saved)
        self.encoder_probed.connect(self._on_probed)
        self.license_status.connect(self._on_license)

        self.ffmpeg = find_ffmpeg()
        self.encoder = str(self.store.value("encoder", "") or "")
        if self.encoder not in ("libx264", "h264_qsv"):
            self.encoder = "libx264"
            threading.Thread(
                target=lambda: self.encoder_probed.emit(probe_encoder(self.ffmpeg)),
                daemon=True).start()

        self.license = LicenseGate(
            self.store, lambda st, msg: self.license_status.emit(st, msg))
        self.license.start()

        self._build_toolbars()
        self._apply_settings(self.cfg, undoable=False)
        self._set_cam(True)
        self.statusBar().showMessage("Checking licence...")

        if self.cfg.get("startup", True):
            QTimer.singleShot(200, self._open_settings)

    # ---------------------------------------------------------- licence
    def _on_license(self, status, message):
        locked = status == "locked"
        self.board.locked = locked
        if locked:
            if self.recording:
                self.stop_recording()           # stop cleanly; already-saved files are kept
            self.lock_overlay.show_message(message)
            self.statusBar().showMessage("This software is locked.")
        else:
            self.lock_overlay.hide()
            self.statusBar().showMessage("Ready.")
        self.rec_btn.setEnabled(not locked)
        self.pause_btn.setEnabled(not locked and self.recording)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.lock_overlay.isVisible():
            self.lock_overlay.setGeometry(self.board.rect())

    # ---------------------------------------------------------- settings
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
        for k in ("bg", "pen", "corner", "size", "width", "startup"):
            s.setValue(k, self.cfg[k])
        s.sync()

    def _apply_settings(self, cfg, undoable=True):
        if QColor(cfg["bg"]) != self.board.bg:
            self.board.set_background(cfg["bg"], undoable=undoable)
        self.board.pen_color = QColor(cfg["pen"])
        self.board.base_width = float(cfg["width"])
        self.board.set_cam(cfg["corner"] if cfg["corner"] in CAM_CORNERS else None,
                           cfg["size"] if cfg["size"] in CAM_SIZES else None)
        self.corner_btn.setText(f"Corner: {self.board.cam_corner}")
        self.camsize_btn.setText("Cam size: " + "SML"[CAM_SIZES.index(self.board.cam_diam)])
        self.width_slider.blockSignals(True)
        self.width_slider.setValue(int(self.board.base_width))
        self.width_slider.blockSignals(False)

    def _open_settings(self):
        if self.board.locked:
            return
        dlg = SettingsDialog(self, self.cfg)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.cfg = dlg.values()
            self._save_settings()
            self._apply_settings(self.cfg)
            self.statusBar().showMessage("Settings saved.")

    def _on_probed(self, enc):
        self.encoder = enc
        self.store.setValue("encoder", enc)
        self.store.sync()

    # ----------------------------------------------------------- toolbars
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

        self.cam_btn = QPushButton("Camera")
        self.cam_btn.setCheckable(True)
        self.cam_btn.setChecked(True)
        self.cam_btn.toggled.connect(self._set_cam)
        tb.addWidget(self.cam_btn)

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

    # ------------------------------------------------------------- webcam
    def _set_cam(self, on):
        if on:
            self.board.cam_on = True
            self.board.cam.start()
            self.ui_timer.start(66)             # ~15fps preview refresh, matches CAM_FPS
            self.cam_watch.start(800)
            self._update_cam_label()
        else:
            self.board.cam.stop()
            self.board.cam_on = False
            self.ui_timer.stop()
            self.cam_watch.stop()
            self.cam_status.setText("  Camera: Off  ")
            self.cam_status.setStyleSheet("font-weight:bold;padding:4px 10px;")
            self.board.touch(self.board._cam_rect())

    def _refresh_cam(self):
        cam = self.board.cam
        if cam.seq != self._seen_cam_seq:
            self._seen_cam_seq = cam.seq
            self.board.update(self.board._to_widget_rect(self.board._cam_rect()))

    def _update_cam_label(self):
        if self.board.cam.state == "on":
            self.cam_status.setText("  Camera: On  ")
            self.cam_status.setStyleSheet("font-weight:bold;color:#2e7d32;padding:4px 10px;")
        else:
            self.cam_status.setText("  Camera: waiting…  ")
            self.cam_status.setStyleSheet("font-weight:bold;color:#c62828;padding:4px 10px;")

    def _cycle_corner(self):
        i = (CAM_CORNERS.index(self.board.cam_corner) + 1) % len(CAM_CORNERS)
        self.board.set_cam(corner=CAM_CORNERS[i])
        self.corner_btn.setText(f"Corner: {CAM_CORNERS[i]}")
        self.cfg["corner"] = CAM_CORNERS[i]
        self._save_settings()

    def _cycle_cam_size(self):
        i = (CAM_SIZES.index(self.board.cam_diam) + 1) % len(CAM_SIZES)
        self.board.set_cam(diam=CAM_SIZES[i])
        self.camsize_btn.setText("Cam size: " + "SML"[i])
        self.cfg["size"] = CAM_SIZES[i]
        self._save_settings()

    # ---------------------------------------------------------- recording
    @property
    def recording(self):
        return self.rec is not None

    def toggle_recording(self):
        if self.board.locked:
            return
        self.stop_recording() if self.recording else self.start_recording()

    def start_recording(self):
        if self.board.locked:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUT_DIR, f"lecture_{stamp}")
        rec = Recorder(base, self.ffmpeg, self.encoder,
                       lambda msg, hw: self.finished.emit(msg, hw))
        try:
            rec.start()
        except Exception as ex:
            self.statusBar().showMessage(f"Could not start recording: {ex}")
            return
        self.rec = rec
        if rec.stream is None:
            self.statusBar().showMessage(
                f"Audio unavailable: {rec.audio_error} Recording video only.")
        else:
            self.statusBar().showMessage(
                f"Recording ({'Quick Sync' if rec.encoder == 'h264_qsv' else 'software'} "
                f"encoder, mic {rec.rate} Hz).")
        self.paused = False
        self.paused_total = 0.0
        self._frame_sig = None
        self._frame_bytes = None
        self.t0 = time.perf_counter()
        self.rec_timer.start(int(1000 / FPS))
        self.clock.start(500)
        self.rec_btn.setText("Stop Recording")
        self.rec_btn.setStyleSheet(STOP_STYLE)
        self.pause_btn.setEnabled(True)

    def _active_elapsed(self):
        now = time.perf_counter()
        cur_pause = (now - self.pause_started) if self.paused else 0.0
        return now - self.t0 - self.paused_total - cur_pause

    def toggle_pause(self):
        if not self.recording:
            return
        if not self.paused:
            self.paused = True
            self.rec.paused = True
            self.pause_started = time.perf_counter()
            self.pause_btn.setText("Resume")
            self.statusBar().showMessage("Recording paused - nothing is being captured.")
        else:
            self.paused_total += time.perf_counter() - self.pause_started
            self.paused = False
            self.rec.paused = False
            self.pause_btn.setText("Pause")
            self.statusBar().showMessage("Recording resumed.")

    def _grab_frame(self):
        rec = self.rec
        if rec is None or self.paused:
            return
        due = int(self._active_elapsed() * FPS) - rec.frames
        if due <= 0:
            return
        sig = (self.board.version, self.board.cam.seq if self.board.cam_on else -1)
        if sig != self._frame_sig or self._frame_bytes is None:
            img = self.board.render_frame()
            self._frame_bytes = img.constBits().asstring(img.sizeInBytes())
            self._frame_sig = sig
        rec.push(self._frame_bytes, min(due, FPS * 2))

    def _tick_label(self):
        s = int(self._active_elapsed())
        if self.paused:
            self.timer_label.setText(f"  PAUSED {s // 60:02d}:{s % 60:02d}  ")
        else:
            self.timer_label.setText(f"  {s // 60:02d}:{s % 60:02d}  ")

    def stop_recording(self):
        rec, self.rec = self.rec, None
        self.rec_timer.stop()
        self.clock.stop()
        rec.stop()
        self.workers.append(rec.vthread)
        self._frame_bytes = None
        self.paused = False
        self.rec_btn.setText("Start Recording")
        self.rec_btn.setStyleSheet(REC_STYLE)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.timer_label.setText("  00:00  ")
        self.statusBar().showMessage("Finishing video, please wait...")

    def _on_saved(self, msg, hw_failed):
        if hw_failed:
            self._on_probed("libx264")
            self.encoder = "libx264"
        self.statusBar().showMessage(msg)

    def closeEvent(self, e):
        self.license.stop()
        if self.recording:
            self.stop_recording()
        self.board.cam.stop()
        self.ui_timer.stop()
        self.cam_watch.stop()
        self.statusBar().showMessage("Finishing video, please wait...")
        QApplication.processEvents()
        for t in self.workers:
            t.join(timeout=300)
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setWindowTitle("Physics Software - Made by NikilPranav")
    win.show()
    sys.exit(app.exec())
