import io
import re
import requests
from collections import deque
from typing import Tuple

from PIL import Image, ImageFilter
from fastapi import FastAPI, Form
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI()

# -------------------------------
# Free-tier safety limits
# -------------------------------
MAX_SIDE = 2500           # raise if you want; higher may risk RAM on Render Free
MAX_PIXELS = 4_000_000    # ~4MP cap to avoid OOM


# -------------------------------
# Color helpers
# -------------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ValueError("bg_hex must be a valid 6-digit hex color like #FFFFFF")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def is_near_color(px_rgb: Tuple[int, int, int], target_rgb: Tuple[int, int, int], tol: int) -> bool:
    r, g, b = px_rgb
    tr, tg, tb = target_rgb
    return abs(r - tr) <= tol and abs(g - tg) <= tol and abs(b - tb) <= tol


def maybe_downscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    return img


# -------------------------------
# Request model
# -------------------------------
class RemoveBgRequest(BaseModel):
    image_url: str

    bg_hex: str = Field(default="#FFFFFF", description="Background key color in hex, e.g. #FFFFFF or #00FF00")
    color_tolerance: int = Field(default=40, ge=0, le=255, description="Per-channel tolerance")

    erode_px: int = Field(default=1, ge=0, le=6, description="Shrinks alpha edge to remove halos")

    # Optional: memory-safe hole removal
    remove_holes: bool = Field(default=False, description="Remove enclosed background-colored holes inside artwork")
    min_hole_area: int = Field(default=400, ge=0, le=10_000_000, description="Only remove holes >= this pixel area")

    # Optional: edge soften (anti-alias)
    edge_soften: bool = Field(default=False, description="Apply a gentle edge soften to reduce jagged edges")
    edge_soften_px: float = Field(default=0.8, ge=0.0, le=5.0, description="Blur radius used for edge soften (default 0.8)")


# -------------------------------
# Core: border-connected background mask (memory-safe 1D)
# -------------------------------
def floodfill_border_background_mask_1d(img_rgb: Image.Image, target_rgb: Tuple[int, int, int], tol: int) -> bytearray:
    w, h = img_rgb.size
    pix = img_rgb.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y):
        return y * w + x

    def try_seed(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_color(pix[x, y], target_rgb, tol):
            mask[i] = 1
            q.append((x, y))

    # seed borders
    for x in range(w):
        try_seed(x, 0)
        try_seed(x, h - 1)
    for y in range(h):
        try_seed(0, y)
        try_seed(w - 1, y)

    # BFS
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                i = idx(nx, ny)
                if mask[i] == 0 and is_near_color(pix[nx, ny], target_rgb, tol):
                    mask[i] = 1
                    q.append((nx, ny))

    return mask


def apply_alpha_from_mask_1d(img_rgb: Image.Image, bg_mask_1d: bytearray) -> Image.Image:
    w, h = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    p = rgba.load()

    i = 0
    for y in range(h):
        for x in range(w):
            if bg_mask_1d[i] == 1:
                r, g, b, _ = p[x, y]
                p[x, y] = (r, g, b, 0)
            i += 1

    return rgba


# -------------------------------
# Optional: memory-safe hole remover (1D visited + BFS)
# -------------------------------
def remove_enclosed_holes_memory_safe(
    img_rgb: Image.Image,
    rgba: Image.Image,
    border_bg_mask_1d: bytearray,
    target_rgb: Tuple[int, int, int],
    tol: int,
    min_area: int,
) -> Image.Image:
    """
    Removes background-colored regions that are NOT connected to border (holes),
    but only if region size >= min_area. Memory-safe using 1D bytearrays.
    """
    if min_area <= 0:
        min_area = 1

    w, h = img_rgb.size
    pix = img_rgb.load()

    visited = bytearray(w * h)  # 0/1
    out = rgba.copy()
    p_out = out.load()

    def idx(x, y):
        return y * w + x

    def is_bg(x, y):
        return is_near_color(pix[x, y], target_rgb, tol)

    for y0 in range(h):
        for x0 in range(w):
            i0 = idx(x0, y0)
            if visited[i0] == 1:
                continue
            visited[i0] = 1

            # only consider bg-colored pixels that are NOT border-connected bg
            if border_bg_mask_1d[i0] == 1:
                continue
            if not is_bg(x0, y0):
                continue

            # BFS region
            q = deque([(x0, y0)])
            region = [i0]

            while q:
                x, y = q.popleft()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        ii = idx(nx, ny)
                        if visited[ii] == 1:
                            continue
                        visited[ii] = 1

                        # skip border-connected background
                        if border_bg_mask_1d[ii] == 1:
                            continue

                        if is_bg(nx, ny):
                            q.append((nx, ny))
                            region.append(ii)

            # apply if large enough
            if len(region) >= min_area:
                for ii in region:
                    x = ii % w
                    y = ii // w
                    r, g, b, _ = p_out[x, y]
                    p_out[x, y] = (r, g, b, 0)

    return out


# -------------------------------
# Erode alpha (halo killer)
# -------------------------------
def erode_alpha_inplace(rgba: Image.Image, erode_px: int) -> Image.Image:
    if erode_px <= 0:
        return rgba

    w, h = rgba.size
    px = rgba.load()

    a = bytearray(w * h)
    i = 0
    for y in range(h):
        for x in range(w):
            a[i] = 1 if px[x, y][3] > 0 else 0
            i += 1

    def idx(x, y):
        return y * w + x

    for _ in range(erode_px):
        to_clear = []
        for y in range(h):
            for x in range(w):
                ii = idx(x, y)
                if a[ii] == 1:
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h and a[idx(nx, ny)] == 0:
                            to_clear.append(ii)
                            break
        for ii in to_clear:
            a[ii] = 0

    out = rgba.copy()
    p2 = out.load()
    i = 0
    for y in range(h):
        for x in range(w):
            r, g, b, _ = p2[x, y]
            p2[x, y] = (r, g, b, 255 if a[i] == 1 else 0)
            i += 1

    return out


# -------------------------------
# Optional: edge soften (anti-alias look)
# -------------------------------
def edge_soften_alpha(rgba: Image.Image, radius: float) -> Image.Image:
    """
    Softens alpha edges using a tiny blur on alpha only, then re-thresholds.
    This reduces jagged edges and green/white pixel leftovers on contours.
    """
    if radius <= 0:
        return rgba

    r, g, b, a = rgba.split()
    a_blur = a.filter(ImageFilter.GaussianBlur(radius=radius))

    # Re-threshold to keep crisp edges, but smoother than pure binary
    # 128 is a good default; can be adjusted later if needed.
    a_thr = a_blur.point(lambda v: 255 if v >= 128 else 0)

    return Image.merge("RGBA", (r, g, b, a_thr))


# -------------------------------
# Pipeline
# -------------------------------
def process_remove_bg(req: RemoveBgRequest) -> bytes:
    r = requests.get(req.image_url, timeout=45)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = maybe_downscale(img)

    target_rgb = hex_to_rgb(req.bg_hex)

    border_bg = floodfill_border_background_mask_1d(img, target_rgb, req.color_tolerance)
    rgba = apply_alpha_from_mask_1d(img, border_bg)

    # Optional holes
    if req.remove_holes:
        rgba = remove_enclosed_holes_memory_safe(
            img_rgb=img,
            rgba=rgba,
            border_bg_mask_1d=border_bg,
            target_rgb=target_rgb,
            tol=req.color_tolerance,
            min_area=req.min_hole_area,
        )

    # Erode (halo killer)
    rgba = erode_alpha_inplace(rgba, req.erode_px)

    # Optional edge soften
    if req.edge_soften:
        rgba = edge_soften_alpha(rgba, req.edge_soften_px)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------
# API endpoints
# -------------------------------
@app.post("/remove-bg")
def remove_bg(req: RemoveBgRequest):
    png_bytes = process_remove_bg(req)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoint": "/remove-bg",
        "test_page": "/test",
        "notes": "Use bg_hex=#FFFFFF for white, #00FF00 for neon green."
    }


# -------------------------------
# Test UI
# -------------------------------
@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 860px; margin: 40px auto; line-height: 1.4;">
        <h2>Background Remover Test</h2>
        <p>Paste an image URL (JPG/PNG). Choose the key background color (#FFFFFF or #00FF00).</p>

        <form method="post" action="/test" style="padding:16px; border:1px solid #ddd; border-radius:10px;">
          <label><b>Image URL:</b></label><br/>
          <input name="image_url" style="width:100%; padding:10px;" placeholder="https://..."><br/><br/>

          <label><b>Background key color (HEX):</b></label><br/>
          <input name="bg_hex" style="width:220px; padding:10px;" value="#FFFFFF">
          <span style="margin-left:10px; color:#555;">Try #00FF00 if you generated a neon green background</span>
          <br/><br/>

          <label><b>Color tolerance:</b> (JPG: 45–65, PNG: 10–35)</label><br/>
          <input name="color_tolerance" type="number" value="55" min="0" max="255" style="width:120px; padding:10px;"><br/><br/>

          <label><b>Erode edge (px):</b> (halo killer; try 1–2)</label><br/>
          <input name="erode_px" type="number" value="1" min="0" max="6" style="width:120px; padding:10px;"><br/><br/>

          <hr style="margin:18px 0;"/>

          <label><b>Remove enclosed holes</b> (optional):</label>
          <input name="remove_holes" type="checkbox">
          <span style="color:#555;">Removes enclosed background pockets inside artwork</span>
          <br/><br/>

          <label><b>Min hole area (px):</b> (try 250–1500)</label><br/>
          <input name="min_hole_area" type="number" value="400" min="0" max="10000000" style="width:140px; padding:10px;"><br/><br/>

          <hr style="margin:18px 0;"/>

          <label><b>Edge soften</b> (optional):</label>
          <input name="edge_soften" type="checkbox" checked>
          <span style="color:#555;">Smooths jagged contour edges</span>
          <br/><br/>

          <label><b>Edge soften radius (px):</b> (default 0.8)</label><br/>
          <input name="edge_soften_px" type="number" value="0.8" step="0.1" min="0" max="5" style="width:140px; padding:10px;"><br/><br/>

          <button type="submit" style="padding:12px 16px; font-weight:bold; cursor:pointer;">
            Generate Transparent PNG
          </button>
        </form>

        <p style="margin-top:16px; color:#666;">
          Tips:
          <br/>• If you see 1px halos, increase <b>Erode</b> to 2.
          <br/>• If green background leaves pixels, increase <b>Tolerance</b> slightly, then rely on <b>Edge soften</b>.
          <br/>• If internal background pockets remain, enable <b>Remove enclosed holes</b> and raise <b>Min hole area</b>.
        </p>
      </body>
    </html>
    """


@app.post("/test")
def test_submit(
    image_url: str = Form(...),
    bg_hex: str = Form("#FFFFFF"),
    color_tolerance: int = Form(55),
    erode_px: int = Form(1),
    remove_holes: str = Form(None),
    min_hole_area: int = Form(400),
    edge_soften: str = Form(None),
    edge_soften_px: float = Form(0.8),
):
    remove_holes_bool = str(remove_holes).lower() in ("true", "1", "on", "yes")
    edge_soften_bool = str(edge_soften).lower() in ("true", "1", "on", "yes")

    req = RemoveBgRequest(
        image_url=image_url,
        bg_hex=bg_hex,
        color_tolerance=int(color_tolerance),
        erode_px=int(erode_px),
        remove_holes=remove_holes_bool,
        min_hole_area=int(min_hole_area),
        edge_soften=edge_soften_bool,
        edge_soften_px=float(edge_soften_px),
    )

    png_bytes = process_remove_bg(req)
    return Response(content=png_bytes, media_type="image/png")
