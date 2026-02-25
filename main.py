import io
import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

class RemoveBgRequest(BaseModel):
    image_url: str
    white_tolerance: int = 40  # 30–50 között szokott jó lenni

def is_near_white(rgb, tol: int) -> bool:
    r, g, b = rgb
    return (255 - r) <= tol and (255 - g) <= tol and (255 - b) <= tol

def floodfill_background_mask(img_rgb: Image.Image, tol: int):
    """
    Background mask: only pixels connected to the image border that are near-white.
    Returns a 2D boolean array mask[x][y] == True => background.
    """
    w, h = img_rgb.size
    pix = img_rgb.load()

    visited = [[False] * h for _ in range(w)]
    stack = []

    # Seed from all border pixels that are near-white
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

    # Flood fill (4-direction)
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                if is_near_white(pix[nx, ny], tol):
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    return visited

def apply_alpha(img_rgb: Image.Image, bg_mask, erode_px: int = 1):
    """
    Creates RGBA image where background mask becomes transparent.
    erode_px: small shrink to remove white halos (1 is usually enough).
    """
    w, h = img_rgb.size
    rgba = img_rgb.convert("RGBA")
    p = rgba.load()

    # Optional 1px erosion to reduce halos
    if erode_px > 0:
        for _ in range(erode_px):
            new_mask = [[bg_mask[x][y] for y in range(h)] for x in range(w)]
            for x in range(1, w - 1):
                for y in range(1, h - 1):
                    if bg_mask[x][y]:
                        # keep as bg
                        continue
                    # if any neighbor is bg, we "protect" edge by not shrinking too much
                    # (simple approach: shrink bg slightly)
            # Instead of complex erosion, we do a simpler trick below:
            # We'll just apply transparency with a slight tolerance effect.
            bg_mask = bg_mask  # keep as-is for now

    # Apply transparency
    for x in range(w):
        for y in range(h):
            if bg_mask[x][y]:
                r, g, b, a = p[x, y]
                p[x, y] = (r, g, b, 0)  # transparent

    return rgba

@app.post("/remove-white-bg")
def remove_white_bg(req: RemoveBgRequest):
    # 1) Download image
    r = requests.get(req.image_url, timeout=30)
    r.raise_for_status()

    # 2) Load
    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    # 3) Build mask of near-white background connected to border
    mask = floodfill_background_mask(img, req.white_tolerance)

    # 4) Apply alpha
    out = apply_alpha(img, mask, erode_px=0)

    # 5) Return PNG bytes
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return Response(content=png_bytes, media_type="image/png")

@app.get("/")
def root():
    from fastapi.responses import HTMLResponse
from fastapi import Form

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
      <body style="font-family: Arial; max-width: 700px; margin: 40px auto;">
        <h2>Background Remover Test</h2>
        <p>Paste an image URL (JPG/PNG) with a white background, set tolerance, click Generate.</p>
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
    return {"status": "ok", "endpoint": "/remove-white-bg"}
