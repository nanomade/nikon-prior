# core/logbook.py
"""
Per-sample HTML session logbook.  Always active while a sample is open.

Usage (called from any module)
------------------------------
    import core.logbook as lb

    lb.open_for_sample(sample, users_root)   # call on sample open/change
    if lb.get():
        lb.get().log('navigate', 'Flake #3', detail='X +1.234  Y -5.678')
        lb.get().log('capture', '20×', detail='img001.png',
                     image_rel='images/img001.png')
    lb.close()                               # call on app exit
"""

import datetime
import os
from pathlib import Path

_ICONS = {
    'navigate':     '→',
    'capture':      '⬛',
    'recipe':       '▶',
    'heat':         '🌡',
    'approach':     '↓',
    'sample':       '◆',
    'note':         '✎',
    'autofocus':    '⬡',
    'registration': '⊕',
    'flake':        '◈',
    'scan':         '▦',
    'mag':          '⊙',
    'index':        '✛',
}

_CSS = """\
* { box-sizing: border-box; }
body { background:#181818; color:#bbb; font:12px/1.5 monospace;
       margin:0; padding:16px 20px; }
h2   { color:#555; font-size:13px; font-weight:normal; margin:0 0 10px;
       border-bottom:1px solid #2a2a2a; padding-bottom:6px; }
.e   { display:flex; gap:8px; padding:4px 0 4px 10px;
       border-left:3px solid #2a2a2a; margin:2px 0; }
.ts  { color:#444; width:60px; flex-shrink:0; padding-top:1px; }
.ic  { width:18px; flex-shrink:0; text-align:center; }
.bd  { flex:1; min-width:0; }
.ti  { color:#ccc; }
.dt  { color:#666; font-size:11px; margin-top:1px; word-break:break-all; }
.th  { display:block; margin-top:4px; max-width:200px;
       border:1px solid #2a2a2a; }
.ev-navigate     { border-color:#2a6e3f; }
.ev-capture      { border-color:#2a4a7a; }
.ev-recipe       { border-color:#7a5a00; }
.ev-heat         { border-color:#7a2a00; }
.ev-approach     { border-color:#4a2a7a; }
.ev-sample       { border-color:#444; }
.ev-note         { border-color:#444; }
.ev-autofocus    { border-color:#1a5a6a; }
.ev-registration { border-color:#2a6a6a; }
.ev-flake        { border-color:#3a6a2a; }
.ev-scan         { border-color:#3a3a6a; }
.ev-mag          { border-color:#3a3a3a; }
.ev-index        { border-color:#5a4a2a; }
"""


class Logbook:
    """Append-mode HTML logbook for one sample directory."""

    def __init__(self, html_path: Path, sample_name: str):
        self._path = html_path
        self._dir  = html_path.parent
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

        if html_path.exists():
            # Append a session-resume divider to the existing file.
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            self._write(
                f'\n<div class="e ev-sample">'
                f'<div class="ts">{ts}</div>'
                f'<div class="ic">◆</div>'
                f'<div class="bd"><div class="ti">── session resumed ──</div>'
                f'</div></div>\n'
            )
        else:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(
                f'<!DOCTYPE html><html><head>\n'
                f'<meta charset="utf-8">\n'
                f'<title>Logbook — {_esc(sample_name)}</title>\n'
                f'<style>{_CSS}</style>\n'
                f'</head><body>\n'
                f'<h2>Logbook · {_esc(sample_name)} · {_esc(now)}</h2>\n',
                encoding='utf-8',
            )

    def log(self, event_type: str, title: str, detail: str = '',
            image_rel: str = '') -> None:
        """Append one timeline entry.

        Parameters
        ----------
        event_type : str
            Determines left-border colour and icon.
            One of: navigate, capture, recipe, heat, approach, sample, note.
        title : str
            Bold summary line.
        detail : str
            Secondary info shown smaller below the title.
        image_rel : str
            Path to an image *relative to the logbook directory*.
            Rendered as a thumbnail if provided.
        """
        ts   = datetime.datetime.now().strftime('%H:%M:%S')
        icon = _ICONS.get(event_type, '·')
        cls  = f'ev-{event_type}'
        detail_html = (f'<div class="dt">{_esc(detail)}</div>' if detail else '')
        img_html    = (f'<img class="th" src="{_esc(image_rel)}">' if image_rel else '')
        self._write(
            f'<div class="e {cls}">'
            f'<div class="ts">{ts}</div>'
            f'<div class="ic">{icon}</div>'
            f'<div class="bd">'
            f'<div class="ti">{_esc(title)}</div>'
            f'{detail_html}{img_html}'
            f'</div></div>\n'
        )

    def rel(self, abs_path: str) -> str:
        """Convert an absolute file path to one relative to the logbook."""
        try:
            return os.path.relpath(abs_path, self._dir)
        except ValueError:
            return abs_path

    def _write(self, html: str) -> None:
        try:
            with open(self._path, 'a', encoding='utf-8') as f:
                f.write(html)
        except Exception as exc:
            print(f'[logbook] write failed: {exc}')


# ── Module-level singleton ────────────────────────────────────────────────────

_current: 'Logbook | None' = None


def open_for_sample(sample: dict | None, users_root) -> None:
    """Open (or re-open) the logbook for *sample*.  Pass None to deactivate."""
    global _current
    if sample is None:
        _current = None
        return
    sdir = Path(str(users_root)) / sample['user'] / sample['folder']
    name = sample.get('name', sample.get('folder', 'unknown'))
    _current = Logbook(sdir / 'logbook.html', name)
    _current.log('sample', f'Sample open · {name}',
                 detail=f"user: {sample.get('user', '?')}")


def get() -> 'Logbook | None':
    """Return the active Logbook, or None if no sample is open."""
    return _current


def close() -> None:
    """Deactivate the logbook (called on app exit)."""
    global _current
    _current = None


def _esc(s) -> str:
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))
