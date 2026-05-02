"""Screenshot capture using Quartz on macOS.

Resolves the display to capture in this order:

1. The active display for the frontmost app's largest normal (layer 0) window,
   by intersecting ``kCGWindowBounds`` with ``CGDisplayBounds`` (same coordinate
   space as the window server — not ``NSScreen.frame``, which can differ).
2. The display under the mouse, using ``CGEventGetLocation`` (same space as
   ``CGDisplayBounds`` — not ``NSEvent.mouseLocation``, which is Cocoa-flipped).
3. ``CGMainDisplayID()`` as a last resort.

Compresses to JPEG under the configured size limit and returns base64-encoded
bytes for the vision API.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Optional

from friday import config

log = logging.getLogger(__name__)


def capture_focused_display() -> Optional[str]:
    """Capture the best-matching display and return base64-encoded JPEG string.

    Returns None if capture fails (e.g. screen recording permission denied).
    """
    t0 = time.monotonic()
    try:
        img = _capture_via_quartz()
    except Exception as exc:
        log.error("Screenshot capture failed: %s", exc)
        return None

    if img is None:
        return None

    b64 = _compress_to_b64(img)
    elapsed_ms = (time.monotonic() - t0) * 1000
    log.debug("Screenshot captured in %.0f ms (%d KB)", elapsed_ms, len(b64) * 3 // 4 // 1024)
    return b64


def _active_display_ids() -> tuple[int, ...]:
    import Quartz

    err, _, count = Quartz.CGGetActiveDisplayList(0, None, None)
    if err != Quartz.kCGErrorSuccess or count == 0:
        return (Quartz.CGMainDisplayID(),)
    err2, displays, _ = Quartz.CGGetActiveDisplayList(count, None, None)
    if err2 != Quartz.kCGErrorSuccess or not displays:
        return (Quartz.CGMainDisplayID(),)
    return tuple(int(x) for x in displays)


def _screencapture_argv_for_display(display_id: int) -> list[str]:
    """Return extra argv fragment for ``screencapture`` (e.g. ``-D`` ``2``).

    ``man screencapture``: -D <n> where 1 is main, 2 secondary, matching the
    order returned by ``CGGetActiveDisplayList``. Falls back to ``-m`` if the
    id is not in the active list.
    """
    ids = _active_display_ids()
    for i, did in enumerate(ids):
        if did == display_id:
            return ["-D", str(i + 1)]
    return ["-m"]


def _cg_rect_from_window_bounds(bounds: dict[str, Any]) -> Any:
    import Quartz

    def gv(key: str) -> float:
        v = bounds.get(key)
        if v is None:
            return 0.0
        return float(v)

    return Quartz.CGRectMake(gv("X"), gv("Y"), gv("Width"), gv("Height"))


def _display_id_for_frontmost_window() -> Optional[int]:
    """Largest on-screen layer-0 window of the frontmost app → display by intersection."""
    import Quartz
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None
    pid = app.processIdentifier()
    if pid == os.getpid():
        return None

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)
    if not window_list:
        return None

    best_area = 0.0
    best_rect = None
    for entry in window_list:
        if entry.get(Quartz.kCGWindowOwnerPID) != pid:
            continue
        if entry.get(Quartz.kCGWindowLayer, 0) != 0:
            continue
        name = entry.get(Quartz.kCGWindowName) or ""
        owner = entry.get(Quartz.kCGWindowOwnerName) or ""
        if name == "Desktop" and owner == "Finder":
            continue
        bounds = entry.get(Quartz.kCGWindowBounds)
        if not bounds:
            continue
        rect = _cg_rect_from_window_bounds(bounds)
        a = rect.size.width * rect.size.height
        if a > best_area:
            best_area = a
            best_rect = rect

    if best_rect is None or best_area < 1.0:
        return None

    best_display: Optional[int] = None
    best_inter = 0.0
    for did in _active_display_ids():
        db = Quartz.CGDisplayBounds(did)
        inter = Quartz.CGRectIntersection(db, best_rect)
        ia = inter.size.width * inter.size.height
        if ia > best_inter:
            best_inter = ia
            best_display = did

    if best_display is not None and best_inter > 0:
        log.debug("Screenshot display from frontmost window intersection: %s", best_display)
        return best_display
    return None


def _display_id_for_mouse() -> Optional[int]:
    """Display whose ``CGDisplayBounds`` contains ``CGEventGetLocation``."""
    import Quartz

    ev = Quartz.CGEventCreate(None)
    if ev is None:
        return None
    loc = Quartz.CGEventGetLocation(ev)
    for did in _active_display_ids():
        if Quartz.CGRectContainsPoint(Quartz.CGDisplayBounds(did), loc):
            log.debug("Screenshot display from mouse location: %s", did)
            return int(did)
    return None


def _resolve_capture_display_id() -> int:
    import Quartz

    for resolver in (_display_id_for_frontmost_window, _display_id_for_mouse):
        did = resolver()
        if did is not None:
            return int(did)
    main = Quartz.CGMainDisplayID()
    log.debug("Screenshot display fallback: CGMainDisplayID %s", main)
    return int(main)


def _capture_via_quartz():
    """Use Quartz (CoreGraphics) to capture the resolved display.

    Falls back to screencapture CLI if pyobjc is unavailable.
    """
    try:
        import Quartz
        from PIL import Image

        display_id = _resolve_capture_display_id()

        image_ref = Quartz.CGDisplayCreateImage(display_id)
        if image_ref is None:
            log.warning("CGDisplayCreateImage returned None — check Screen Recording permission")
            return None

        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)

        bitmapData = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image_ref))
        if bitmapData is None:
            return None

        import ctypes

        buf = (ctypes.c_uint8 * len(bitmapData)).from_buffer_copy(bytes(bitmapData))
        img = Image.frombuffer("RGBA", (width, height), bytes(buf), "raw", "BGRA", 0, 1)
        img = img.convert("RGB")
        return img

    except ImportError:
        log.warning("pyobjc not available, falling back to screencapture CLI")
        return _capture_via_cli()


def _capture_via_cli():
    """Fallback: use macOS ``screencapture`` CLI tool."""
    import subprocess
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name

    extra: list[str] = ["-m"]
    try:
        did = _resolve_capture_display_id()
        extra = _screencapture_argv_for_display(did)
    except ImportError:
        pass

    result = subprocess.run(
        ["screencapture", "-x", *extra, tmp_path],
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        log.error("screencapture failed: %s", result.stderr)
        return None

    img = Image.open(tmp_path).convert("RGB")
    import os as os_mod

    os_mod.unlink(tmp_path)
    return img


def _compress_to_b64(img) -> str:
    """Compress PIL Image to JPEG under config.SCREENSHOT_MAX_KB, return base64."""
    from PIL import Image

    # Downscale if very large — GPT-4o doesn't benefit from >1920px wide
    max_width = 1920
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    max_bytes = config.SCREENSHOT_MAX_KB * 1024
    quality = config.SCREENSHOT_JPEG_QUALITY

    for _ in range(10):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes or quality <= 20:
            break
        quality -= 10

    return base64.b64encode(data).decode("utf-8")
