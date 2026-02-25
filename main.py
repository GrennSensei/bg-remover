import io
import requests
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel

app = FastAPI()


# -------------------------------
# Request model (API használathoz)
# -------------------------------

class RemoveBgRequest(BaseModel):
    image_url: str
    white_tolerance: int = 40


# -------------------------------
# Segédfüggvények
# -------------------------------

def is_near_white(rgb, tol: int) -> bool:
    r, g, b = rgb
    return (255 - r) <= tol and (255 - g) <= tol and (255 - b) <= tol


def floodfill_background_mask(img_rgb: Image.Image, tol: int):
    """
    Csak a kép széléhez kapcsolódó near-white pixeleket tekintjük háttérnek.
    """
    w, h = img_rgb.size
    pix = img_rgb.load()

    visited = [[False] * h for _ in range(w)]
    stack = []

    # Seed: border pixelek
    for x in range(w):
        for y in (0, h - 1):
            if not visited[x][y] and is_near_white(pix[x, y], tol):
                visited[x][y] = True
                stack.append((x, y))

    for y in range(h):
        for x in (0, w - 1):
            if not visited[x][y] and is_near_white(pix[x, y], tol):
                visited[x][y] = True
                stack.append((x, y))

    # Flood fill (4 irány)
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                if is_near_white(pix[nx, ny], tol):
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    return visited


def apply_alpha(img_rgb: Image.Image, bg_mask):
    w, h = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    p = rgba.load()

    for x in range(w):
        for y in range(h):
            if bg_mask[x][y]:
                r, g, b, a = p[x, y]
                p[x, y] = (r, g, b, 0)

    return rgba


# -------------------------------
# API endpoint (JSON alapú)
# -------------------------------

@app.post("/remove-white-bg")
def remove_white_bg(req: RemoveBgRequest):
    # 1. Kép letöltése
    r = requests.get(req.image_url, timeout=30)
    r.raise_for_status()

    # 2. Kép betöltése
    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    # 3. Maszk készítése
    mask = floodfill_background_mask(img, req.white_tolerance)

    # 4. Alpha alkalmazása
    out = apply_alpha(img, mask)

    # 5. PNG visszaküldése
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return Response(content=png_bytes, media_type="image/png")


# -------------------------------
# Egyszerű teszt weboldal
# -------------------------------

@app.get("/")
def root():
    return {"status": "ok", "endpoint": "/remove-white-bg", "test_page": "/test"}


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h2>Background Remover Test</h2>
        <p>Paste an image URL (JPG/PNG) with a white background.</p>
        <form method="post" action="/test">
          <label>Image URL:</label><br/>
          <input name="image_url" style="width:100%;" placeholder="https://..."><br/><br/>
          <label>White tolerance (JPG: 45–65):</label><br/>
          <input name="white_tolerance" type="number" value="55"><br/><br/>
          <button type="submit">Generate Transparent PNG</button>
        </form>
      </body>
    </html>
    """


@app.post("/test")
def test_submit(image_url: str = Form(...), white_tolerance: int = Form(55)):
    req = RemoveBgRequest(image_url=image_url, white_tolerance=white_tolerance)
    return remove_white_bg(req)
