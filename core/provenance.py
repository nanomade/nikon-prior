"""Provenance stamps for derived artifacts (plan task S1, born inside L4).

Every derived file (contrast calibrations, detection outputs, maps) should be
traceable to the code, inputs, and parameters that produced it. This helper
returns a small dict to embed under a 'provenance' key; keep it cheap enough
to call from any writer.

The per-sample optical store (core.sample_data.write_optical_calibration)
already stamps git rev + date in its meta.json — this generalises that
pattern for everything else.
"""
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_HASH_LIMIT = 8 * 1024 * 1024   # sha1 files up to 8 MB; mtime+size above


def git_rev() -> str | None:
    """Current commit (short) + '-dirty' if the tree has changes; None outside git."""
    try:
        repo = Path(__file__).resolve().parent.parent
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=repo, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        if not rev:
            return None
        dirty = subprocess.run(['git', 'status', '--porcelain'],
                               cwd=repo, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return rev + ('-dirty' if dirty else '')
    except Exception:
        return None


def _describe_input(path: Path) -> dict:
    try:
        st = path.stat()
        d = {'size': st.st_size, 'mtime': datetime.fromtimestamp(
            st.st_mtime).isoformat(timespec='seconds')}
        if st.st_size <= _HASH_LIMIT:
            d['sha1'] = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        return d
    except OSError:
        return {'missing': True}


def provenance_stamp(inputs: dict | None = None, params: dict | None = None) -> dict:
    """Build a provenance dict: git rev, UTC timestamp, input digests, params.

    inputs: {label: path} — each stamped with size/mtime and sha1 when small.
    params: any JSON-serialisable dict of the parameters that mattered.
    """
    stamp = {
        'git_rev': git_rev(),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    if inputs:
        stamp['inputs'] = {str(k): _describe_input(Path(v))
                           for k, v in inputs.items()}
    if params:
        stamp['params'] = params
    return stamp
