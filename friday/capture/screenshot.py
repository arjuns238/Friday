"""Screenshot capture using Quartz/ScreenCaptureKit on macOS.

Captures the display containing the currently focused window,
compresses to JPEG under the configured size limit, and returns
base64-encoded bytes ready for the OpenAI vision API.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

from friday import config

log = logging.getLogger(__name__)


def capture_focused_display() -> Optional[str]:
    """Capture the focused display and return base64-encoded JPEG string.

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


def _capture_via_quartz():
    """Use Quartz (CoreGraphics) to capture the main display.

    Falls back to screencapture CLI if pyobjc is unavailable.
    """
    try:
        import Quartz
        from AppKit import NSScreen
        from PIL import Image

        # Determine which display to capture: the one with the key window.
        # For simplicity, capture the main display (index 0).
        # Phase 4 can improve this to detect the focused display.
        display_id = Quartz.CGMainDisplayID()

        image_ref = Quartz.CGDisplayCreateImage(display_id)
        if image_ref is None:
            log.warning("CGDisplayCreateImage returned None — check Screen Recording permission")
            return None

        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)

        # Convert CGImage → PIL Image via bitmap data
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
    """Fallback: use macOS `screencapture` CLI tool."""
    import subprocess
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name

    result = subprocess.run(
        ["screencapture", "-x", "-m", tmp_path],
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        log.error("screencapture failed: %s", result.stderr)
        return None

    img = Image.open(tmp_path).convert("RGB")
    import os
    os.unlink(tmp_path)
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
