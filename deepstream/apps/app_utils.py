import os
import sys
import logging
import socket
import yaml
import cv2
import threading
import termios
import tty
import enum
import numpy as np

# --- GStreamer imports ---
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import GstRtspServer
    RTSP_AVAILABLE = True
except Exception:
    GstRtspServer = None
    RTSP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# nvdsosd RGBA colours used for OSD overlays
COLOR_LOCKED    = (0.0, 1.0, 0.5, 1.0)    # Teal  — active track
COLOR_CANDIDATE = (1.0, 1.0, 1.0, 1.0)    # White — selection highlight
COLOR_DETECTION = (0.5, 0.5, 0.5, 0.7)    # Grey  — background detections
COLOR_SEARCHING = (1.0, 0.5, 0.0, 1.0)    # Orange — re-ID scan
COLOR_STALE     = (1.0, 0.2, 0.2, 0.9)    # Red   — lost/stale

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

def load_yaml(path):
    """Load and parse a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(level_str):
    """Initialize logging with the specified level."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')

# ---------------------------------------------------------------------------
# Raw Terminal Input ("tty" keystroke reader)
# ---------------------------------------------------------------------------

class KeyReader:
    """
    Non-blocking single-keypress reader for Linux terminals.
    Reads one character at a time without requiring the Enter key.

    Usage:
        reader = KeyReader()
        reader.start()    # spawns daemon thread
        # Later in main loop:
        key = reader.get()   # returns latest char or None
        reader.stop()
    """

    def __init__(self):
        self._latest  = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name='key-reader')
        self._thread.start()

    def stop(self):
        self._running = False

    def get(self):
        """Return the latest key pressed (and clear it), or None."""
        with self._lock:
            k = self._latest
            self._latest = None
            return k

    def _run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if ch:
                    with self._lock:
                        self._latest = ch
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ---------------------------------------------------------------------------
# Math & Geometry
# ---------------------------------------------------------------------------

def get_iou(boxA, boxB):
    """Calculate IoU between two boxes [x, y, w, h]."""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, x2 - x1)
    interH = max(0, y2 - y1)
    interArea = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0
    return interArea / float(unionArea)

# ---------------------------------------------------------------------------
# RTSP & Network
# ---------------------------------------------------------------------------

def get_local_ip():
    """Try to determine the local primary IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def setup_rtsp_server(rtsp_port, udp_port, mount_path='/sutrack'):
    """Create GstRtspServer instance."""
    if not RTSP_AVAILABLE or GstRtspServer is None:
        return None

    server  = GstRtspServer.RTSPServer.new()
    server.props.service = str(rtsp_port)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc port=%d '
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" ! '
        'rtph264depay ! rtph264pay name=pay0 pt=96 )' % udp_port
    )
    factory.set_shared(True)

    server.get_mount_points().add_factory(mount_path, factory)
    server.attach(None)
    return server


# ---------------------------------------------------------------------------
# Phase 10 — State & Re-ID Utilities
# ---------------------------------------------------------------------------

class TrackingState(enum.Enum):
    IDLE      = 'IDLE'
    SELECTING = 'SELECTING'   # OSD cycling mode
    LOCKED    = 'LOCKED'
    SEARCHING = 'SEARCHING'
    STALE     = 'STALE'


def compute_crop_histogram(frame_rgb, bbox):
    """H-S histogram of an RGB crop for appearance-based Re-ID."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x = max(0, x); y = max(0, y)
    crop = frame_rgb[y:y+h, x:x+w]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compare_histograms(hist1, hist2):
    """Bhattacharyya-based similarity. Returns 0..1 (1 = identical)."""
    if hist1 is None or hist2 is None:
        return 0.0
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - float(dist))


class IDHistory:
    """Stores appearance signature of a tracked target for re-ID after re-entry."""
    def __init__(self):
        self.template_hist   = None
        self.last_bbox       = None   # [x,y,w,h] last known position
        self.last_frame_seen = -1
        self.init_bbox       = None   # original selection bbox

    def update(self, frame_rgb, bbox, frame_idx):
        hist = compute_crop_histogram(frame_rgb, bbox)
        if hist is not None:
            self.template_hist = hist
        self.last_bbox       = list(bbox)
        self.last_frame_seen = frame_idx

    def match_score(self, frame_rgb, bbox):
        """Returns 0..1 similarity of a candidate crop vs stored template."""
        if self.template_hist is None:
            return 0.0
        return compare_histograms(self.template_hist,
                                  compute_crop_histogram(frame_rgb, bbox))

    def clear(self):
        self.template_hist   = None
        self.last_bbox       = None
        self.last_frame_seen = -1
        self.init_bbox       = None
