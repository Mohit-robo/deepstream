"""
SUTrack V5 Remote Client (Phase 12)
GTK3 client that receives the RTSP stream from the Jetson server
and sends REST commands for tracker control.

Requirements (local machine):
    sudo apt install python3-gi python3-gi-cairo gstreamer1.0-tools \\
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \\
        gstreamer1.0-plugins-bad gstreamer1.0-libav

Usage:
    python deepstream/apps/v5_remote_client.py --host 192.168.1.100
    python deepstream/apps/v5_remote_client.py --host 192.168.1.100 \\
        --rtsp-port 8554 --api-port 8000

Controls:
    s = Select     n = Next      p = Prev
    l / Enter = Lock             q / Esc = Cancel      x = Quit
    Left-click on video = Click-to-lock nearest detection on the server
"""

import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gst, GLib, Gtk, Gdk

try:
    import cairo
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False

# ---------------------------------------------------------------------------
# GTK3 Client Window
# ---------------------------------------------------------------------------

class SUTrackClient(Gtk.Window):
    """
    GTK3 window that:
      1. Receives the Jetson RTSP stream via a GStreamer appsink pipeline.
      2. Renders frames on a DrawingArea using Cairo (RGB -> BGRA -> ImageSurface).
      3. Dispatches REST commands on button clicks, keyboard presses, and
         mouse clicks on the video canvas.
      4. Polls GET /api/state every 0.5 s to refresh the FPS / State labels.

    Normalised click coordinates
    ----------------------------
    When the user clicks on the video, we compute the pixel offset within the
    rendered video rectangle (_draw_rect), divide by its dimensions, and send
    normalised (x, y) in [0.0, 1.0] to the server's POST /api/command click
    endpoint.  The server maps those coordinates to its live frame dimensions
    and selects the nearest detection centroid — fully latency-insensitive.
    """

    def __init__(self, args):
        super().__init__(title='SUTrack V5 — Remote Client')
        self.args = args
        self.log  = logging.getLogger('client')

        self._api_base = 'http://%s:%d' % (args.host, args.api_port)
        self._rtsp_url = 'rtsp://%s:%d%s' % (args.host, args.rtsp_port, args.rtsp_path)

        # Latest decoded frame (numpy RGB H x W x 3); written by GStreamer thread.
        self._frame      = None
        self._frame_lock = threading.Lock()

        # Video render rectangle within the DrawingArea: (ox, oy, dw, dh).
        # Written in _on_draw (GTK thread); read in _on_canvas_click (GTK thread).
        self._draw_rect = (0.0, 0.0, float(args.width), float(args.height))

        self._running   = True
        self._pipeline  = None

        self._build_ui()
        self._start_gst_pipeline()
        self._start_state_poll()

        self.show_all()

    # ── UI Layout ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.set_default_size(self.args.width, self.args.height + 90)
        self.connect('destroy',         self._on_close)
        self.connect('key-press-event', self._on_key)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(vbox)

        # Video canvas — Cairo renders RTSP frames here
        self.canvas = Gtk.DrawingArea()
        self.canvas.set_size_request(self.args.width, self.args.height)
        self.canvas.override_background_color(
            Gtk.StateFlags.NORMAL, Gdk.RGBA(0, 0, 0, 1))
        self.canvas.connect('draw', self._on_draw)
        self.canvas.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.canvas.connect('button-press-event', self._on_canvas_click)
        vbox.pack_start(self.canvas, True, True, 0)

        # Info bar: FPS and tracking state (polled from server)
        info_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        info_bar.set_margin_start(12)
        info_bar.set_margin_end(12)
        info_bar.set_margin_top(4)
        info_bar.set_margin_bottom(2)
        vbox.pack_start(info_bar, False, False, 0)

        self.fps_label   = Gtk.Label(label='FPS: --')
        self.state_label = Gtk.Label(label='State: --')
        self.fps_label.set_xalign(0)
        self.state_label.set_xalign(0)
        info_bar.pack_start(self.fps_label,         False, False, 0)
        info_bar.pack_start(Gtk.Label(label=' | '), False, False, 0)
        info_bar.pack_start(self.state_label,       False, False, 0)

        # Control buttons
        btn_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_bar.set_margin_start(12)
        btn_bar.set_margin_end(12)
        btn_bar.set_margin_top(4)
        btn_bar.set_margin_bottom(8)
        vbox.pack_start(btn_bar, False, False, 0)

        def make_btn(label, action, css=None):
            b = Gtk.Button(label=label)
            b.connect('clicked', lambda _: self._cmd(action))
            if css:
                b.get_style_context().add_class(css)
            btn_bar.pack_start(b, True, True, 0)
            return b

        make_btn('Select', 'select')
        make_btn('Prev',   'prev')
        make_btn('Next',   'next')
        make_btn('Lock',   'lock',   'suggested-action')
        make_btn('Cancel', 'cancel', 'destructive-action')

        css = Gtk.CssProvider()
        css.load_from_data(b"""
            window { background-color: #1c1c1c; }
            label  { color: #e0e0e0; font-size: 13px; }
            button { font-size: 13px; min-height: 34px; }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    # ── GStreamer pipeline (receive RTSP, decode to RGB, push to appsink) ──

    def _start_gst_pipeline(self):
        # Use rtspsrc with protocols=tcp to force RTP-over-TCP (interleaved).
        # This avoids the UDP firewall/NAT issue where rtspsrc negotiates a UDP
        # media port that is blocked between the Jetson and the local PC.
        # rtspsrc has dynamic pads; parse_launch auto-links them to rtph264depay.
        pipe_str = (
            'rtspsrc name=src location=%s protocols=tcp latency=200 ! '
            'rtph264depay ! '
            'h264parse ! '
            'avdec_h264 ! '
            'videoconvert ! '
            'video/x-raw,format=RGB ! '
            'appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false'
        ) % self._rtsp_url

        self._pipeline = None
        try:
            self._pipeline = Gst.parse_launch(pipe_str)
            appsink = self._pipeline.get_by_name('appsink')
            appsink.connect('new-sample', self._on_new_sample)
            self.log.debug('Using rtspsrc+tcp pipeline.')
        except Exception as e:
            self.log.warning('parse_launch failed (%s); falling back to manual pipeline.', e)
            self._pipeline = None

        if self._pipeline is None:
            self._pipeline = self._build_manual_pipeline()

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::eos',   self._on_gst_eos)
        bus.connect('message::error', self._on_gst_error)

        self.log.info('Connecting to RTSP: %s', self._rtsp_url)
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.log.error('GStreamer pipeline failed to start — check RTSP URL.')

    def _build_manual_pipeline(self):
        """Fallback: rtspsrc(tcp) with explicit pad-added linking."""
        pipeline = Gst.Pipeline.new('client')

        src     = Gst.ElementFactory.make('rtspsrc',      'src')
        depay   = Gst.ElementFactory.make('rtph264depay', 'depay')
        parse   = Gst.ElementFactory.make('h264parse',    'parse')
        dec     = Gst.ElementFactory.make('avdec_h264',   'dec')
        vconv   = Gst.ElementFactory.make('videoconvert', 'vconv')
        capsf   = Gst.ElementFactory.make('capsfilter',   'caps')
        appsink = Gst.ElementFactory.make('appsink',      'appsink')

        src.set_property('location',  self._rtsp_url)
        src.set_property('protocols', 0x4)   # GST_RTSP_LOWER_TRANS_TCP = 4
        src.set_property('latency',   200)
        capsf.set_property('caps', Gst.Caps.from_string('video/x-raw,format=RGB'))
        appsink.set_property('emit-signals', True)
        appsink.set_property('max-buffers',  1)
        appsink.set_property('drop',         True)
        appsink.set_property('sync',         False)
        appsink.connect('new-sample', self._on_new_sample)

        for el in [src, depay, parse, dec, vconv, capsf, appsink]:
            pipeline.add(el)

        depay.link(parse)
        parse.link(dec)
        dec.link(vconv)
        vconv.link(capsf)
        capsf.link(appsink)

        def on_pad_added(element, pad):
            caps_str = pad.query_caps(None).to_string()
            if 'x-rtp' in caps_str or 'video' in caps_str:
                sink = depay.get_static_pad('sink')
                if not sink.is_linked():
                    pad.link(sink)

        src.connect('pad-added', on_pad_added)
        self.log.debug('Using manual rtspsrc+tcp pipeline.')
        return pipeline

    # ── appsink callback (GStreamer streaming thread) ─────────────────────

    def _on_new_sample(self, appsink):
        """Called from the GStreamer streaming thread for each decoded frame."""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        ok_w, w = s.get_int('width')
        ok_h, h = s.get_int('height')
        if not (ok_w and ok_h):
            return Gst.FlowReturn.OK

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mapinfo)

        with self._frame_lock:
            self._frame = frame

        # Schedule a GTK redraw on the main thread
        GLib.idle_add(self.canvas.queue_draw)
        return Gst.FlowReturn.OK

    # ── Cairo frame rendering (GTK main thread) ───────────────────────────

    def _on_draw(self, widget, cr):
        """Render the latest RTSP frame onto the DrawingArea using Cairo."""
        alloc = widget.get_allocation()
        cw, ch = alloc.width, alloc.height

        # Always fill background black first
        cr.set_source_rgb(0, 0, 0)
        cr.paint()

        if not CAIRO_AVAILABLE:
            # pycairo missing — show install hint
            cr.set_source_rgb(1.0, 0.4, 0.4)
            cr.select_font_face('monospace')
            cr.set_font_size(16)
            cr.move_to(20, ch / 2)
            cr.show_text('Install python3-gi-cairo to enable video rendering')
            return

        with self._frame_lock:
            frame = self._frame

        if frame is None:
            # Show connecting status until first frame arrives
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.select_font_face('monospace',
                                cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(16)
            cr.move_to(20, ch / 2)
            cr.show_text('Connecting to %s ...' % self._rtsp_url)
            return

        h, w = frame.shape[:2]

        # Aspect-fit: scale the frame to fill the canvas, centred
        scale = min(cw / w, ch / h)
        dw    = w * scale
        dh    = h * scale
        ox    = (cw - dw) / 2.0
        oy    = (ch - dh) / 2.0

        # Store rendered video rect for click normalisation (GTK thread only)
        self._draw_rect = (ox, oy, dw, dh)

        # Convert RGB numpy (H, W, 3) -> BGRA bytes.
        # Cairo FORMAT_ARGB32 on little-endian stores pixels as [B, G, R, A] in memory.
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[:, :, 0] = frame[:, :, 2]   # B
        bgra[:, :, 1] = frame[:, :, 1]   # G
        bgra[:, :, 2] = frame[:, :, 0]   # R
        bgra[:, :, 3] = 255              # A (fully opaque)

        stride  = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
        surface = cairo.ImageSurface.create_for_data(
            bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, w, h, stride)

        cr.translate(ox, oy)
        cr.scale(scale, scale)
        cr.set_source_surface(surface, 0, 0)
        cr.get_source().set_filter(cairo.FILTER_BILINEAR)
        cr.paint()

    # ── Mouse click on video → REST click command ─────────────────────────

    def _on_canvas_click(self, widget, event):
        """Map a left-click on the video canvas to normalised coords and POST."""
        if event.button != 1:
            return
        ox, oy, dw, dh = self._draw_rect
        if dw <= 0 or dh <= 0:
            return
        # Pixel offset within the rendered video rectangle
        vx = event.x - ox
        vy = event.y - oy
        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, vx / dw))
        ny = max(0.0, min(1.0, vy / dh))
        self.log.info('Click -> normalised=(%.3f, %.3f)', nx, ny)
        self._cmd('click', x=round(nx, 4), y=round(ny, 4))

    # ── Keyboard shortcuts → REST commands ───────────────────────────────

    def _on_key(self, _widget, event):
        k = event.keyval
        if   k in (Gdk.KEY_s, Gdk.KEY_S):                      self._cmd('select')
        elif k == Gdk.KEY_n:                                    self._cmd('next')
        elif k == Gdk.KEY_p:                                    self._cmd('prev')
        elif k in (Gdk.KEY_l, Gdk.KEY_L, Gdk.KEY_Return):     self._cmd('lock')
        elif k in (Gdk.KEY_q, Gdk.KEY_Q, Gdk.KEY_Escape):     self._cmd('cancel')
        elif k in (Gdk.KEY_x, Gdk.KEY_X):                      self._on_close(None)

    # ── Fire-and-forget HTTP POST ─────────────────────────────────────────

    def _cmd(self, action, **kwargs):
        """Dispatch a REST command in a daemon thread (non-blocking)."""
        payload = {'action': action}
        payload.update(kwargs)
        url = self._api_base + '/api/command'

        def _send():
            try:
                body = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    url, data=body,
                    headers={'Content-Type': 'application/json'},
                    method='POST')
                with urllib.request.urlopen(req, timeout=2.0):
                    pass
                self.log.debug('CMD %s ok', action)
            except Exception as e:
                self.log.warning('CMD %s failed: %s', action, e)

        threading.Thread(target=_send, daemon=True, name='cmd-%s' % action).start()

    # ── State polling (updates FPS / State labels) ────────────────────────

    def _start_state_poll(self):
        """Polls GET /api/state every 0.5 s; updates labels via GLib.idle_add."""
        url = self._api_base + '/api/state'

        def _poll():
            while self._running:
                try:
                    with urllib.request.urlopen(url, timeout=2.0) as resp:
                        data = json.loads(resp.read())
                    GLib.idle_add(self._update_labels, data)
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=_poll, daemon=True, name='state-poll').start()

    def _update_labels(self, data):
        """Called on the GTK main thread via GLib.idle_add."""
        self.fps_label.set_text(  'FPS: %.1f'  % data.get('fps',   0.0))
        self.state_label.set_text('State: %s'  % data.get('state', '?'))
        return False   # do not repeat

    # ── GStreamer bus callbacks ────────────────────────────────────────────

    def _on_gst_eos(self, bus, msg):
        self.log.info('RTSP stream ended (EOS) — retrying in 5 s...')
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        GLib.timeout_add_seconds(5, self._reconnect)

    def _on_gst_error(self, bus, msg):
        err, dbg = msg.parse_error()
        self.log.error('GStreamer error: %s | %s', err, dbg)
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        self.log.info('Retrying in 5 s...')
        GLib.timeout_add_seconds(5, self._reconnect)

    def _reconnect(self):
        """Tear down the current pipeline and reconnect to RTSP (GTK main thread)."""
        if not self._running:
            return False
        self.log.info('Reconnecting to RTSP: %s', self._rtsp_url)
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        with self._frame_lock:
            self._frame = None          # triggers "Connecting..." text on canvas
        GLib.idle_add(self.canvas.queue_draw)
        self._start_gst_pipeline()
        return False                    # do not repeat the timeout

    # ── Window close ─────────────────────────────────────────────────────

    def _on_close(self, _widget):
        self._running = False
        self.log.info('Closing client.')
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SUTrack V5 Remote Client')
    parser.add_argument('--host',       required=True,
                        help='Jetson server IP or hostname')
    parser.add_argument('--rtsp-port',  type=int, default=8554,
                        help='RTSP server port on the Jetson (default: 8554)')
    parser.add_argument('--rtsp-path',  default='/sutrack',
                        help='RTSP mount path (default: /sutrack)')
    parser.add_argument('--api-port',   type=int, default=8000,
                        help='REST API port on the Jetson (default: 8000)')
    parser.add_argument('--width',      type=int, default=1280,
                        help='Initial canvas width in pixels')
    parser.add_argument('--height',     type=int, default=720,
                        help='Initial canvas height in pixels')
    parser.add_argument('--log-level',  default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')
    log = logging.getLogger('main')

    if not CAIRO_AVAILABLE:
        log.warning(
            'pycairo not installed — video will not render. '
            'Fix: sudo apt install python3-gi-cairo')

    log.info('Server: %s  RTSP:%d  API:%d  Path:%s',
             args.host, args.rtsp_port, args.api_port, args.rtsp_path)

    Gst.init(None)
    SUTrackClient(args)
    Gtk.main()


if __name__ == '__main__':
    main()
