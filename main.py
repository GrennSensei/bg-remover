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
# Config: free-tier safety limits
# -------------------------------
MAX_SIDE = 2000          # safety downscale (keeps tests stable on free tier)
MAX_PIXELS = 3_000_000   # hard cap (~3MP) to avoid OOM


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        raise ValueError("bg_hex must be #RRGGBB")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def is_near_color(px_rgb: Tuple[int, int, int], target_rgb: Tuple[int, int, int], tol: int) -> bool:
    r, g, b = px_rgb
    tr, tg, tb = target_rgb
    return abs(r - tr) <= tol and abs(g - tg) <= tol and abs(b - tb) <= tol


class RemoveBgRequest(BaseModel):
    image_url: str
    bg_hex: str = Field(default="#FFFFFF")
    color_tolerance: int = Field(default=40, ge=0, le=255)
    erode_px: int = Field(default=1, ge=0, le=6)
    # NOTE: holes OFF by default on free tier; can enable if images are small
    remove_holes: bool = Field(default=False)
    min_hole_area: int = Field(default=400, ge=0, le=10_000_000)


def maybe_downscale(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w * h > MAX_PIXELS:
        # downscale so that w*h <= MAX_PIXELS
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    # also cap the longest side for stability
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    return img


def floodfill_border_background_mask_1d(img_rgb: Image.Image, target_rgb: Tuple[int, int, int], tol: int) -> bytearray:
    """
    Memory-safe: returns 1D bytearray mask length w*h
    mask[i] = 1 if background connected to border
    """
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

    # seeds: borders
    for x in range(w):
        try_seed(x, 0)
        try_seed(x, h - 1)
    for y in range(h):
        try_seed(0, y)
        try_seed(w - 1, y)

    # BFS floodfill 4-neighbor
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
                r, g, b, a = p[x, y]
                p[x, y] = (r, g, b, 0)
            i += 1

    return rgba


def erode_alpha_inplace(rgba: Image.Image, erode_px: int) -> Image.Image:
    if erode_px <= 0:
        return rgba

    w, h = rgba.size
    px = rgba.load()

    # alpha as 1D bytearray
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
                i = idx(x, y)
                if a[i] == 1:
                    # if any neighbor is transparent -> clear
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= nx < w and 0 <= ny < h:
                            if a[idx(nx, ny)] == 0:
                                to_clear.append(i)
                                break
        for i in to_clear:
            a[i] = 0

    # apply alpha back
    i = 0
    out = rgba.copy()
    p2 = out.load()
    for y in range(h):
        for x in range(w):
            r, g, b, _ = p2[x, y]
            p2[x, y] = (r, g, b, 255 if a[i] == 1 else 0)
            i += 1

    return out


def process_remove_bg(req: RemoveBgRequest) -> bytes:
    r = requests.get(req.image_url, timeout=45)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = maybe_downscale(img)

    target_rgb = hex_to_rgb(req.bg_hex)

    bg_mask = floodfill_border_background_mask_1d(img, target_rgb, req.color_tolerance)
    rgba = apply_alpha_from_mask_1d(img, bg_mask)

    # NOTE: remove_holes is intentionally disabled by default for free tier stability.
    # If you really need it later, we can implement a low-memory version too.

    rgba = erode_alpha_inplace(rgba, req.erode_px)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/remove-bg")
def remove_bg(req: RemoveBgRequest):
    png_bytes = process_remove_bg(req)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "/remove-bg", "test_page": "/test"}


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 820px; margin: 40px auto; line-height: 1.4;">
        <h2>Background Remover Test (Free-tier safe)</h2>
        <p>Paste an image URL (JPG/PNG). Choose the background key color you used (#FFFFFF or #00FF00).</p>

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
          <input name="erode_px" type="number" value="1" min="0" max="6" style="width:120px; padding:10px;"><br/><br/>

          <p style="color:#777; margin:0 0 12px 0;">
            Note: "Remove enclosed holes" is disabled on Render Free to prevent out-of-memory crashes.
            If you need it, we can add a memory-safe version later.
          </p>

          <button type="submit" style="padding:12px 16px; font-weight:bold; cursor:pointer;">
            Generate Transparent PNG
          </button>
        </form>

        <p style="margin-top:16px; color:#666;">
          Tip: If a 1px outline remains, increase <b>Erode edge</b> to 2.
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
):
    req = RemoveBgRequest(
        image_url=image_url,
        bg_hex=bg_hex,
        color_tolerance=int(color_tolerance),
        erode_px=int(erode_px),
        remove_holes=False,
        min_hole_area=0,
    )
    png_bytes = process_remove_bg(req)
    return Response(content=png_bytes, media_type="image/png")
