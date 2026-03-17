from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial Bold.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _build_icon_base(size: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (size, size), "#070B10")
    draw = ImageDraw.Draw(img)

    for y in range(size):
        blend = y / max(1, size - 1)
        r = int((7 * (1.0 - blend)) + (11 * blend))
        g = int((11 * (1.0 - blend)) + (18 * blend))
        b = int((16 * (1.0 - blend)) + (32 * blend))
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    inset = int(size * 0.1)
    draw.rounded_rectangle(
        [inset, inset, size - inset, size - inset],
        radius=int(size * 0.18),
        fill="#0E1626",
        outline="#00E5FF",
        width=max(6, int(size * 0.012)),
    )
    draw.rounded_rectangle(
        [inset + int(size * 0.06), inset + int(size * 0.06), size - inset - int(size * 0.06), size - inset - int(size * 0.06)],
        radius=int(size * 0.12),
        fill="#121C2F",
        outline="#00FF66",
        width=max(4, int(size * 0.008)),
    )

    text = "ST"
    font = _load_font(int(size * 0.42))
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx = (size - tw) / 2.0
    ty = (size - th) / 2.0 - int(size * 0.02)
    draw.text((tx + 3, ty + 3), text, font=font, fill=(0, 0, 0, 180))
    draw.text((tx, ty), text, font=font, fill="#00FF66")

    return img


def ensure_icons(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "super_trader.png"
    ico_path = out_dir / "super_trader.ico"
    icns_path = out_dir / "super_trader.icns"

    base = _build_icon_base(1024)
    base.save(png_path, format="PNG")
    base.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    try:
        base.save(icns_path, format="ICNS")
    except Exception:
        pass

    return {"png": png_path, "ico": ico_path, "icns": icns_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Super Trader icon assets (.png/.ico/.icns).")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "icons"),
        help="Output directory for generated icons.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    icons = ensure_icons(out_dir)
    print(f"[icons] png={icons['png']}")
    print(f"[icons] ico={icons['ico']}")
    if icons["icns"].exists():
        print(f"[icons] icns={icons['icns']}")
    else:
        print("[icons] icns=not generated (ICNS encoder unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
