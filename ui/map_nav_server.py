"""
MapNavServer — tiny HTTP server that accepts POST /navigate {x_mm, y_mm}
from the DZI map viewer and emits a navigate(x_mm, y_mm) Qt signal.

Runs in a daemon QThread.  Port is pre-bound in __init__ so callers can
read .port immediately without waiting for the thread to start.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket

from PyQt5.QtCore import QThread, pyqtSignal

_PREFERRED_PORT = 57373


class MapNavServer(QThread):
    navigate        = pyqtSignal(float, float, object)   # x_mm, y_mm, scan_placement|None
    import_flakes   = pyqtSignal(list, object)    # [{x_mm, y_mm, note}, ...], scan_placement|None
    update_flake    = pyqtSignal(str, object)     # flake_id, {layer_count: N}
    delete_flake    = pyqtSignal(str)             # flake_id
    label_candidate = pyqtSignal(object)          # {scan_folder,id,label,x_mm,y_mm,...}

    current_sample_folder: str = ''             # updated by stagecontrol on sample change

    def __init__(self, parent=None):
        super().__init__(parent)
        # Pre-bind so .port is available immediately (before thread starts)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(('127.0.0.1', _PREFERRED_PORT))
        except OSError:
            self._sock.bind(('127.0.0.1', 0))
        self._sock.listen(5)
        self._port: int = self._sock.getsockname()[1]
        self._server = None

    @property
    def port(self) -> int:
        return self._port

    def run(self):
        sig_nav    = self.navigate
        sig_import = self.import_flakes
        sig_update = self.update_flake
        sig_delete = self.delete_flake
        sig_label  = self.label_candidate
        server_inst = self
        sock = self._sock

        class _Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_POST(self):
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    body = json.loads(self.rfile.read(n))
                    path = self.path.split('?')[0]
                    resp = b'{"ok":true}'
                    if path == '/ping':
                        # Send the LIVE placement so the browser can show
                        # current-stage coords (reference→current) without a map
                        # rebuild on re-register.  Stored flake coords are
                        # reference-stage; current = apply_placement(reference).
                        _pl = None
                        _folder = server_inst.current_sample_folder
                        if _folder:
                            try:
                                with open(os.path.join(_folder, 'sample.json')) as _f:
                                    _reg = ((json.load(_f).get('placement') or {})
                                            .get('registration') or {})
                                if 'dx_mm' in _reg or _reg.get('chip_transform'):
                                    _pl = {'dx_mm': _reg.get('dx_mm', 0.0),
                                           'dy_mm': _reg.get('dy_mm', 0.0),
                                           'rotation_deg': _reg.get('rotation_deg', 0.0)}
                            except Exception:
                                _pl = None
                        resp = json.dumps({'ok': True,
                            'sample_folder': server_inst.current_sample_folder,
                            'placement': _pl,
                        }).encode()
                    elif path == '/navigate':
                        # Optional scan_placement = the placement transform active
                        # when the map's scan was captured; lets the app compose
                        # scan-frame→current correctly (None → legacy behaviour).
                        sig_nav.emit(float(body['x_mm']), float(body['y_mm']),
                                     body.get('scan_placement'))
                    elif path == '/import_flakes':
                        # Accept both the legacy bare list and the new
                        # {markers, scan_placement} envelope.
                        if isinstance(body, dict):
                            sig_import.emit(list(body.get('markers', [])),
                                            body.get('scan_placement'))
                        else:
                            sig_import.emit(list(body), None)
                    elif path == '/update_flake':
                        _EDITABLE = {'layer_count', 'name', 'notes', 'cleanliness',
                                     'isolation', 'magnification', 'area_um2', 'status',
                                     'confirmed'}
                        fields = {k: body[k] for k in _EDITABLE if k in body}
                        sig_update.emit(str(body['id']), fields)
                    elif path == '/delete_flake':
                        sig_delete.emit(str(body['id']))
                    elif path == '/label_candidate':
                        sig_label.emit(dict(body))
                    elif path == '/flakes':
                        folder = server_inst.current_sample_folder
                        if folder:
                            _sj = os.path.join(folder, 'sample.json')
                            try:
                                with open(_sj) as _f:
                                    _s = json.load(_f)
                                resp = json.dumps(_s.get('flakes', [])).encode()
                            except Exception:
                                resp = b'[]'
                        else:
                            resp = b'[]'
                    else:
                        self.send_response(404)
                        self._cors()
                        self.end_headers()
                        return
                    self.send_response(200)
                    self._cors()
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception:
                    self.send_response(400)
                    self._cors()
                    self.end_headers()

            def _cors(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')

            def log_message(self, *args):
                pass

        # bind_and_activate=False: skip HTTPServer's own socket.bind() so it
        # doesn't collide with our pre-bound self._sock on the same port.
        server = HTTPServer(('127.0.0.1', self._port), _Handler,
                            bind_and_activate=False)
        server.socket.close()
        server.socket = sock
        self._server = server
        server.serve_forever()

    def stop(self):
        if self._server:
            self._server.shutdown()
