"""Generate app/photoholic.ico (the GUI's logo mark: gold rounded square with a lens ring).

ai\\.venv\\Scripts\\python.exe packaging\\make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

GOLD = (232, 176, 75, 255)
INK = (20, 21, 24, 255)
OUT = Path(__file__).resolve().parents[1] / "app" / "photoholic.ico"


def draw(size: int = 256) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 32
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=7 * s, fill=GOLD)
    d.rectangle([10 * s, 6 * s, 18 * s, 10 * s], fill=INK)  # camera top
    d.ellipse([9 * s, 10 * s, 23 * s, 24 * s], outline=INK, width=max(1, round(3 * s)))  # lens
    return im


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)]
    draw().save(OUT, sizes=sizes)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
