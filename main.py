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
# Render Free safety limits (avoid OOM)
# -------------------------------
MAX_SIDE = 3000            # you asked max ~3000x3000 for testing
MAX_PIXELS = 6_000_000     # keep conservative; 3000x3000=9M would downscale a bit


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
    # pixel cap first
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return img.resize((nw, nh), Image.LANCZOS)

    # side cap
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

    bg_hex: str = Field(default="#FFFFFF", description="Key background color, e.g. #FFFFFF or #00FF00")
    color_tolerance: int = Field(default=90, ge=0, description="Relaxed tolerance (JPG often needs 45–120)")

    # Two-stage floodfill (recommended for noisy JPG green/white)
    two_stage: bool = Field(default=True, description="Use strict+relaxed floodfill to avoid eating artwork edges")
    strict_tolerance: int = Field(default=35, ge=0, description="Strict tolerance seed for border floodfill")

    erode_px: int = Field(default=1, ge=0, description="Shrinks alpha edge to remove halos")

    # Optional: memory-safe hole removal
    remove_holes: bool = Field(default=True, description="Remove enclosed background-colored holes inside artwork")
    min_hole_area: int = Field(default=400, ge=0, description="Remove holes >= this pixel area")

    # Optional: edge soften (anti-alias)
    edge_soften: bool = Field(default=False, description="Smooth jagged edges (recommended when tolerance is high)")
    edge_soften_px: float = Field(default=1.2, ge=0.0, description="Blur radius for edge soften (default 1.2)")

    # Optional: remove residual neon green pixels in the output
    cleanup_residual_green: bool = Field(
        default=False,
        description="If true, removes remaining #00FF00 (and close) pixels from the output PNG (made transparent).",
    )

    # Adjustable percent for green cleanup tolerance expansion (NO MAX)
    cleanup_green_percent: float = Field(
        default=15.0,
        ge=0.0,
        description="Percent expansion applied to color_tolerance when removing residual #00FF00 pixels (default 15%).",
    )

    # Edge-only cleanup switch
    cleanup_green_edge_only: bool = Field(
        default=True,
        description="If true, green cleanup applies only along alpha edges (recommended).",
    )

    # NEW: adjustable expansion for the edge-only region (NO MAX)
    cleanup_edge_expand_percent: float = Field(
        default=15.0,
        ge=0.0,
        description="Expands the edge-only cleanup region by a % that maps to a few-pixel dilation (default 15%).",
    )


# -------------------------------
# Background mask (1D, memory-safe)
# Two-stage: seed with strict tolerance, then expand with relaxed tolerance
# -------------------------------
def floodfill_border_bg_mask_two_stage_1d(
    img_for_keying: Image.Image,
    target_rgb: Tuple[int, int, int],
    tol_strict: int,
    tol_relaxed: int,
) -> bytearray:
    w, h = img_for_keying.size
    pix = img_for_keying.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y):
        return y * w + x

    def try_seed_strict(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_color(pix[x, y], target_rgb, tol_strict):
            mask[i] = 1
            q.append((x, y))

    # Seed border using STRICT tolerance (safe)
    for x in range(w):
        try_seed_strict(x, 0)
        try_seed_strict(x, h - 1)
    for y in range(h):
        try_seed_strict(0, y)
        try_seed_strict(w - 1, y)

    # Expand from strict seeds using RELAXED tolerance (captures JPG noise/spill),
    # but only through regions connected to border background.
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                i = idx(nx, ny)
                if mask[i] == 0 and is_near_color(pix[nx, ny], target_rgb, tol_relaxed):
                    mask[i] = 1
                    q.append((nx, ny))

    return mask


def floodfill_border_bg_mask_single_stage_1d(
    img_for_keying: Image.Image,
    target_rgb: Tuple[int, int, int],
    tol: int,
) -> bytearray:
    w, h = img_for_keying.size
    pix = img_for_keying.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y):
        return y * w + x

    def try_seed(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_color(pix[x, y], target_rgb, tol):
            mask[i] = 1
            q.append((x, y))

    for x in range(w):
        try_seed(x, 0)
        try_seed(x, h - 1)
    for y in range(h):
        try_seed(0, y)
        try_seed(w - 1, y)

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
# Optional: cleanup residual neon green pixels in already-transparent PNG
# -------------------------------
def cleanup_residual_green_pixels(rgba: Image.Image, tol: int) -> Image.Image:
    w, h = rgba.size
    out = rgba.copy()
    p = out.load()

    target = (0, 255, 0)

    for y in range(h):
        for x in range(w):
            r, g, b, a = p[x, y]
            if a > 0 and is_near_color((r, g, b), target, tol):
                p[x, y] = (r, g, b, 0)

    return out


def build_alpha_edge_mask(rgba: Image.Image) -> bytearray:
    """
    Edge mask = pixels that are on/near the alpha boundary.
    A pixel is considered 'edge' if:
      - 0 < alpha < 255 (semi-transparent), OR
      - alpha > 0 and any 4-neighbor has alpha == 0 (touching transparency)
    """
    w, h = rgba.size
    p = rgba.load()
    mask = bytearray(w * h)

    def idx(x, y):
        return y * w + x

    for y in range(h):
        for x in range(w):
            a = p[x, y][3]
            if 0 < a < 255:
                mask[idx(x, y)] = 1
                continue
            if a > 0:
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        if p[nx, ny][3] == 0:
                            mask[idx(x, y)] = 1
                            break

    return mask


def dilate_mask_4n(mask: bytearray, w: int, h: int, iters: int) -> bytearray:
    """
    Simple 4-neighbor dilation for a 0/1 mask.
    iters = how many pixels outward to expand.
    """
    if iters <= 0:
        return mask

    cur = bytearray(mask)
    for _ in range(iters):
        nxt = bytearray(cur)
        i = 0
        for y in range(h):
            for x in range(w):
                if cur[i] == 1:
                    # set neighbors
                    if x > 0:
                        nxt[i - 1] = 1
                    if x < w - 1:
                        nxt[i + 1] = 1
                    if y > 0:
                        nxt[i - w] = 1
                    if y < h - 1:
                        nxt[i + w] = 1
                i += 1
        cur = nxt

    return cur


def cleanup_residual_green_pixels_edge_only(rgba: Image.Image, tol: int, edge_mask: bytearray) -> Image.Image:
    w, h = rgba.size
    out = rgba.copy()
    p = out.load()

    target = (0, 255, 0)

    i = 0
    for y in range(h):
        for x in range(w):
            if edge_mask[i] == 1:
                r, g, b, a = p[x, y]
                if a > 0 and is_near_color((r, g, b), target, tol):
                    p[x, y] = (r, g, b, 0)
            i += 1

    return out


# -------------------------------
# Optional: memory-safe hole remover (1D visited + BFS)
# -------------------------------
def remove_enclosed_holes_memory_safe(
    img_for_keying: Image.Image,
    rgba: Image.Image,
    border_bg_mask_1d: bytearray,
    target_rgb: Tuple[int, int, int],
    tol: int,
    min_area: int,
) -> Image.Image:
    if min_area <= 0:
        min_area = 1

    w, h = img_for_keying.size
    pix = img_for_keying.load()

    visited = bytearray(w * h)
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

            # Only consider bg-like pixels that are NOT border-connected bg
            if border_bg_mask_1d[i0] == 1:
                continue
            if not is_bg(x0, y0):
                continue

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

                        if border_bg_mask_1d[ii] == 1:
                            continue

                        if is_bg(nx, ny):
                            q.append((nx, ny))
                            region.append(ii)

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
def erode_alpha(rgba: Image.Image, erode_px: int) -> Image.Image:
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
# Optional: edge soften (anti-alias look) - refined (smoother edges)
# -------------------------------
def edge_soften_alpha(rgba: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return rgba

    r, g, b, a = rgba.split()

    # Refined: keep the blurred alpha (no hard threshold), so tiny details are smoother.
    a_blur = a.filter(ImageFilter.GaussianBlur(radius=radius))

    return Image.merge("RGBA", (r, g, b, a_blur))


# -------------------------------
# Pipeline
# -------------------------------
def process_remove_bg(req: RemoveBgRequest) -> bytes:
    r = requests.get(req.image_url, timeout=60)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = maybe_downscale(img)

    # Keying uses a *slightly smoothed* version to reduce JPG speckle in background.
    # Artwork pixels are never modified; only the mask decisions use this.
    img_key = img.filter(ImageFilter.BoxBlur(0.6))

    target_rgb = hex_to_rgb(req.bg_hex)
    tol_relaxed = int(req.color_tolerance)

    # Auto-sane strict tolerance for noisy JPGs (unless user sets something else)
    tol_strict = int(req.strict_tolerance)
    if req.two_stage:
        # keep strict below relaxed; if user accidentally set higher, clamp
        if tol_strict >= tol_relaxed:
            tol_strict = max(10, tol_relaxed - 30)

        border_bg = floodfill_border_bg_mask_two_stage_1d(
            img_for_keying=img_key,
            target_rgb=target_rgb,
            tol_strict=tol_strict,
            tol_relaxed=tol_relaxed,
        )
    else:
        border_bg = floodfill_border_bg_mask_single_stage_1d(
            img_for_keying=img_key,
            target_rgb=target_rgb,
            tol=tol_relaxed,
        )

    rgba = apply_alpha_from_mask_1d(img_rgb=img, bg_mask_1d=border_bg)

    # Optional holes (uses same keying image)
    if req.remove_holes:
        rgba = remove_enclosed_holes_memory_safe(
            img_for_keying=img_key,
            rgba=rgba,
            border_bg_mask_1d=border_bg,
            target_rgb=target_rgb,
            tol=tol_relaxed,
            min_area=req.min_hole_area,
        )

    # Optional: cleanup remaining neon green spill pixels in output (if enabled)
    if req.cleanup_residual_green:
        green_tol = int(req.color_tolerance * (1.0 + (req.cleanup_green_percent / 100.0)))  # NO MAX CAP

        if req.cleanup_green_edge_only:
            w, h = rgba.size
            edge_mask = build_alpha_edge_mask(rgba)

            # "Percent" -> a few pixels of dilation:
            # 0–9% => 0px, 10–19% => 1px, 20–29% => 2px, ...
            expand_px = int(req.cleanup_edge_expand_percent // 10)
            edge_mask = dilate_mask_4n(edge_mask, w, h, expand_px)

            rgba = cleanup_residual_green_pixels_edge_only(rgba, tol=green_tol, edge_mask=edge_mask)
        else:
            rgba = cleanup_residual_green_pixels(rgba, tol=green_tol)

    # Erode edge (halo killer)
    rgba = erode_alpha(rgba, req.erode_px)

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
        "defaults": {
            "bg_hex": "#FFFFFF",
            "color_tolerance": 90,
            "two_stage": True,
            "strict_tolerance": 35,
            "erode_px": 1,
            "edge_soften": False,
            "edge_soften_px": 1.2,
            "remove_holes": True,
            "min_hole_area": 400,
            "cleanup_residual_green": False,
            "cleanup_green_percent": 15.0,
            "cleanup_green_edge_only": True,
            "cleanup_edge_expand_percent": 15.0,
        },
        "tips": [
            "JPG green-screen often needs high tolerance; two_stage helps prevent eating edges.",
            "If green pixels remain: raise color_tolerance first, then enable edge_soften.",
            "If internal background pockets remain: remove_holes is ON by default; adjust min_hole_area as needed.",
            "If neon green spill remains: enable cleanup_residual_green and increase cleanup_green_percent + edge expand.",
        ],
    }


# -------------------------------
# Test UI
# -------------------------------
@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 900px; margin: 40px auto; line-height: 1.4;">
        <h2>Background Remover Test</h2>
        <p>Paste an image URL (JPG/PNG). Choose background: White (#FFFFFF) or Green (#00FF00).</p>

        <form method="post" action="/test" style="padding:16px; border:1px solid #ddd; border-radius:10px;">
          <label><b>Image URL:</b></label><br/>
          <input name="image_url" style="width:100%; padding:10px;" placeholder="https://..."><br/><br/>

          <label><b>Background color:</b></label><br/>
          <label style="margin-right:14px;">
            <input id="bg_green" name="bg_green" type="checkbox">
            #00FF00 (green)
          </label>
          <label>
            <input id="bg_white" name="bg_white" type="checkbox" checked>
            #FFFFFF (white)
          </label>

          <script>
            (function () {
              var green = document.getElementById('bg_green');
              var white = document.getElementById('bg_white');

              function enforceExclusive(changed) {
                if (changed === green && green.checked) white.checked = false;
                if (changed === white && white.checked) green.checked = false;

                // ensure one is always selected
                if (!green.checked && !white.checked) white.checked = true;
              }

              green.addEventListener('change', function () { enforceExclusive(green); });
              white.addEventListener('change', function () { enforceExclusive(white); });

              // initialize
              enforceExclusive(white);
            })();
          </script>

          <br/><br/>

          <label><b>Relaxed tolerance:</b> (JPG often 60–140)</label><br/>
          <input name="color_tolerance" type="number" value="90" min="0" style="width:120px; padding:10px;"><br/><br/>

          <label><b>Two-stage keying (recommended for JPG):</b></label>
          <input name="two_stage" type="checkbox" checked>
          <span style="color:#555;">Strict seeds + relaxed expansion</span>
          <br/><br/>

          <label><b>Strict tolerance:</b> (seed; typically 20–45)</label><br/>
          <input name="strict_tolerance" type="number" value="35" min="0" style="width:120px; padding:10px;"><br/><br/>

          <label><b>Erode edge (px):</b> (halo killer; try 1–2)</label><br/>
          <input name="erode_px" type="number" value="1" min="0" style="width:120px; padding:10px;"><br/><br/>

          <hr style="margin:18px 0;"/>

          <label><b>Remove enclosed holes</b> (default ON):</label>
          <input name="remove_holes" type="checkbox" checked>
          <span style="color:#555;">Removes enclosed background pockets inside artwork</span>
          <br/><br/>

          <label><b>Min hole area (px):</b> (try 250–1500)</label><br/>
          <input name="min_hole_area" type="number" value="400" min="0" style="width:140px; padding:10px;"><br/><br/>

          <hr style="margin:18px 0;"/>

          <label><b>Edge soften</b> (optional):</label>
          <input name="edge_soften" type="checkbox">
          <span style="color:#555;">Smoother tiny details (improved)</span>
          <br/><br/>

          <label><b>Edge soften radius (px):</b></label><br/>
          <input name="edge_soften_px" type="number" value="1.2" step="0.1" min="0.96" style="width:140px; padding:10px;"><br/><br/>

          <hr style="margin:18px 0;"/>

          <label><b>Cleanup residual neon green (#00FF00)</b> (optional):</label>
          <input name="cleanup_residual_green" type="checkbox">
          <span style="color:#555;">Removes leftover green spill pixels from the transparent PNG</span>
          <br/><br/>

          <label><b>Cleanup edge-only</b> (recommended):</label>
          <input name="cleanup_green_edge_only" type="checkbox" checked>
          <span style="color:#555;">Applies cleanup only near alpha edges</span>
          <br/><br/>

          <label><b>Green cleanup tolerance boost (%):</b> (default 15)</label><br/>
          <input name="cleanup_green_percent" type="number" value="15" step="1" min="0"
                 style="width:140px; padding:10px;">
          <br/><br/>

          <label><b>Edge-only expand (%):</b> (default 15)</label><br/>
          <input name="cleanup_edge_expand_percent" type="number" value="15" step="1" min="0"
                 style="width:140px; padding:10px;">
          <span style="margin-left:10px; color:#555;">10% ≈ +1px, 20% ≈ +2px, ...</span>
          <br/><br/>

          <button type="submit" style="padding:12px 16px; font-weight:bold; cursor:pointer;">
            Generate Transparent PNG
          </button>
        </form>

        <p style="margin-top:16px; color:#666;">
          If tiny green specks remain in deep corners:
          enable cleanup + edge-only, then raise
          <b>Edge-only expand</b> to 30–60% and <b>Green boost</b> to 30–120%.
        </p>
      </body>
    </html>
    """


@app.post("/test")
def test_submit(
    image_url: str = Form(...),
    bg_green: str = Form(None),
    bg_white: str = Form(None),
    color_tolerance: int = Form(90),
    two_stage: str = Form(None),
    strict_tolerance: int = Form(35),
    erode_px: int = Form(1),
    remove_holes: str = Form(None),
    min_hole_area: int = Form(400),
    edge_soften: str = Form(None),
    edge_soften_px: float = Form(1.2),
    cleanup_residual_green: str = Form(None),
    cleanup_green_percent: float = Form(15.0),
    cleanup_green_edge_only: str = Form(None),
    cleanup_edge_expand_percent: float = Form(15.0),
):
    def to_bool(v):
        return str(v).lower() in ("true", "1", "on", "yes")

    bg_hex = "#00FF00" if to_bool(bg_green) else "#FFFFFF"

    req = RemoveBgRequest(
        image_url=image_url,
        bg_hex=bg_hex,
        color_tolerance=int(color_tolerance),
        two_stage=to_bool(two_stage),
        strict_tolerance=int(strict_tolerance),
        erode_px=int(erode_px),
        remove_holes=to_bool(remove_holes),
        min_hole_area=int(min_hole_area),
        edge_soften=to_bool(edge_soften),
        edge_soften_px=float(edge_soften_px),
        cleanup_residual_green=to_bool(cleanup_residual_green),
        cleanup_green_percent=float(cleanup_green_percent),
        cleanup_green_edge_only=to_bool(cleanup_green_edge_only),
        cleanup_edge_expand_percent=float(cleanup_edge_expand_percent),
    )

    png_bytes = process_remove_bg(req)
    return Response(content=png_bytes, media_type="image/png")
