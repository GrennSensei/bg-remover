import io
import re
import requests
from collections import deque
from typing import Tuple

from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI()


# -------------------------------
# Helpers: color parsing + distance
# -------------------------------

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Accepts '#RRGGBB' or 'RRGGBB'. Returns (r,g,b).
    """
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ValueError("bg_hex must be a valid 6-digit hex color like #FFFFFF")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def is_near_color(px_rgb: Tuple[int, int, int], target_rgb: Tuple[int, int, int], tol: int) -> bool:
    """
    Simple per-channel tolerance check (fast + robust for solid BG).
    """
    r, g, b = px_rgb
    tr, tg, tb = target_rgb
    return abs(r - tr) <= tol and abs(g - tg) <= tol and abs(b - tb) <= tol


# -------------------------------
# Request model
# -------------------------------

class RemoveBgRequest(BaseModel):
    image_url: str
    bg_hex: str = Field(default="#FFFFFF", description="Background key color in hex, e.g. #FFFFFF or #00FF00")
    color_tolerance: int = Field(default=40, ge=0, le=255, description="Tolerance for background keying (JPG often needs 45–65)")
    erode_px: int = Field(default=1, ge=0, le=10, description="Erode alpha edge to remove halos (0–3 typical)")
    remove_holes: bool = Field(default=True, description="Remove enclosed background-colored holes inside the artwork")
    min_hole_area: int = Field(default=250, ge=0, le=10_000_000, description="Only remove enclosed holes larger than this pixel area")


# -------------------------------
# Core algorithms
# -------------------------------

def floodfill_border_background(img_rgb: Image.Image, target_rgb: Tuple[int, int, int], tol: int):
    """
    Returns a boolean mask [w][h] where True == background connected to border (safe removal).
    Uses 4-neighbor floodfill from image border pixels that match the background color.
    """
    w, h = img_rgb.size
    pix = img_rgb.load()

    bg = [[False] * h for _ in range(w)]
    q = deque()

    # seed border pixels
    def try_seed(x, y):
        if not bg[x][y] and is_near_color(pix[x, y], target_rgb, tol):
            bg[x][y] = True
            q.append((x, y))

    for x in range(w):
        try_seed(x, 0)
        try_seed(x, h - 1)
    for y in range(h):
        try_seed(0, y)
        try_seed(w - 1, y)

    # BFS floodfill
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not bg[nx][ny]:
                if is_near_color(pix[nx, ny], target_rgb, tol):
                    bg[nx][ny] = True
                    q.append((nx, ny))

    return bg


def apply_alpha_from_bgmask(img_rgb: Image.Image, bg_mask):
    """
    Converts background pixels (True in bg_mask) to alpha=0, keeps others alpha=255.
    """
    w, h = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    p = rgba.load()

    for x in range(w):
        for y in range(h):
            if bg_mask[x][y]:
                r, g, b, a = p[x, y]
                p[x, y] = (r, g, b, 0)

    return rgba


def erode_alpha(rgba: Image.Image, erode_px: int):
    """
    Shrinks the opaque region by eroding alpha edges.
    This reduces 1px halos from imperfect background colors.
    """
    if erode_px <= 0:
        return rgba

    w, h = rgba.size
    px = rgba.load()

    # build alpha map
    alpha = [[0] * h for _ in range(w)]
    for x in range(w):
        for y in range(h):
            alpha[x][y] = px[x, y][3]

    # For each iteration, any opaque pixel (alpha>0) that touches a transparent neighbor becomes transparent.
    for _ in range(erode_px):
        to_clear = []
        for x in range(w):
            for y in range(h):
                if alpha[x][y] > 0:
                    # if any 4-neighbor is transparent, clear this pixel
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h:
                            if alpha[nx][ny] == 0:
                                to_clear.append((x, y))
                                break
        for x, y in to_clear:
            alpha[x][y] = 0

    # apply back
    out = rgba.copy()
    p2 = out.load()
    for x in range(w):
        for y in range(h):
            r, g, b, _ = p2[x, y]
            p2[x, y] = (r, g, b, alpha[x][y])

    return out


def remove_enclosed_holes(img_rgb: Image.Image, rgba: Image.Image, target_rgb: Tuple[int, int, int], tol: int, min_area: int):
    """
    Finds regions of background-colored pixels NOT connected to border (i.e., holes inside artwork),
    and makes them transparent IF their area >= min_area.
    This helps remove inner background pockets while avoiding tiny highlights.
    """
    if min_area <= 0:
        min_area = 1

    w, h = img_rgb.size
    pix = img_rgb.load()

    visited = [[False] * h for _ in range(w)]

    def is_bg(x, y):
        return is_near_color(pix[x, y], target_rgb, tol)

    out = rgba.copy()
    p_out = out.load()

    for x0 in range(w):
        for y0 in range(h):
            if visited[x0][y0]:
                continue
            if not is_bg(x0, y0):
                visited[x0][y0] = True
                continue

            # BFS this region
            q = deque([(x0, y0)])
            visited[x0][y0] = True
            region = [(x0, y0)]
            touches_border = (x0 == 0 or y0 == 0 or x0 == w - 1 or y0 == h - 1)

            while q:
                x, y = q.popleft()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                        visited[nx][ny] = True
                        if is_bg(nx, ny):
                            q.append((nx, ny))
                            region.append((nx, ny))
                            if nx == 0 or ny == 0 or nx == w - 1 or ny == h - 1:
                                touches_border = True

            # Only remove if it's enclosed and big enough
            if (not touches_border) and (len(region) >= min_area):
                for x, y in region:
                    r, g, b, a = p_out[x, y]
                    p_out[x, y] = (r, g, b, 0)

    return out


def process_remove_bg(
    image_url: str,
    bg_hex: str,
    color_tolerance: int,
    erode_px: int,
    remove_holes: bool,
    min_hole_area: int
) -> bytes:
    # Download
    r = requests.get(image_url, timeout=45)
    r.raise_for_status()

    # Load as RGB
    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    # Parse target background color
    target_rgb = hex_to_rgb(bg_hex)

    # 1) Safe border-connected background removal
    bg_mask = floodfill_border_background(img, target_rgb, color_tolerance)
    rgba = apply_alpha_from_bgmask(img, bg_mask)

    # 2) Optional enclosed holes removal
    if remove_holes:
        rgba = remove_enclosed_holes(img, rgba, target_rgb, color_tolerance, min_hole_area)

    # 3) Optional alpha edge erosion to kill halos
    rgba = erode_alpha(rgba, erode_px)

    # Output PNG
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------
# API endpoint (general)
# -------------------------------

@app.post("/remove-bg")
def remove_bg(req: RemoveBgRequest):
    png_bytes = process_remove_bg(
        image_url=req.image_url,
        bg_hex=req.bg_hex,
        color_tolerance=req.color_tolerance,
        erode_px=req.erode_px,
        remove_holes=req.remove_holes,
        min_hole_area=req.min_hole_area,
    )
    return Response(content=png_bytes, media_type="image/png")


# -------------------------------
# Simple test web UI
# -------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoint": "/remove-bg",
        "test_page": "/test",
        "tips": "For JPG use color_tolerance 45-65. For halos increase erode_px to 2."
    }


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 820px; margin: 40px auto; line-height: 1.4;">
        <h2>Background Remover Test</h2>
        <p>Paste an image URL (JPG/PNG). Choose the exact background key color you used in generation (e.g. #FFFFFF or #00FF00).</p>

        <form method="post" action="/test" style="padding:16px; border:1px solid #ddd; border-radius:10px;">
          <label><b>Image URL:</b></label><br/>
          <input name="image_url" style="width:100%; padding:10px;" placeholder="https://..."><br/><br/>

          <label><b>Background key color (HEX):</b></label><br/>
          <input name="bg_hex" style="width:220px; padding:10px;" value="#FFFFFF">
          <span style="margin-left:10px; color:#555;">Try #00FF00 if you generated a neon green background</span>
          <br/><br/>

          <label><b>Color tolerance:</b> (JPG: 45–65, PNG: 10–40)</label><br/>
          <input name="color_tolerance" type="number" value="55" min="0" max="255" style="width:120px; padding:10px;"><br/><br/>

          <label><b>Erode edge (px):</b> (halo killer, try 1–2)</label><br/>
          <input name="erode_px" type="number" value="1" min="0" max="10" style="width:120px; padding:10px;"><br/><br/>

          <label><b>Remove enclosed holes:</b></label>
          <input name="remove_holes" type="checkbox" checked>
          <span style="color:#555;">Removes internal background pockets</span><br/><br/>

          <label><b>Min hole area (px):</b> (avoid removing tiny highlights, try 250–1000)</label><br/>
          <input name="min_hole_area" type="number" value="250" min="0" max="10000000" style="width:140px; padding:10px;"><br/><br/>

          <button type="submit" style="padding:12px 16px; font-weight:bold; cursor:pointer;">
            Generate Transparent PNG
          </button>
        </form>

        <p style="margin-top:16px; color:#666;">
          Tip: If a 1px outline remains, increase <b>Erode edge</b> to 2.
          If internal background spots remain, increase <b>Min hole area</b> slightly or keep <b>Remove enclosed holes</b> enabled.
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
    remove_holes: bool = Form(False),
    min_hole_area: int = Form(250),
):
    # Checkbox handling: if present -> True
    # FastAPI Form sends remove_holes as string sometimes; normalize:
    remove_holes_bool = str(remove_holes).lower() in ("true", "1", "on", "yes")

    png_bytes = process_remove_bg(
        image_url=image_url,
        bg_hex=bg_hex,
        color_tolerance=int(color_tolerance),
        erode_px=int(erode_px),
        remove_holes=remove_holes_bool,
        min_hole_area=int(min_hole_area),
    )
    return Response(content=png_bytes, media_type="image/png")
