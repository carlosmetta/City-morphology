"""Pre-render every city's street network as a PNG backdrop for the City map tab.

    python make_citymaps.py

Reads data/geom_*.npz, writes data/citymaps/<city>.png — dark streets on a
transparent ground, so the cluster dots sit on top of a real map instead of
floating in space. Run once after embed_grid.py; takes a few minutes.
"""
import argparse, time
from pathlib import Path

import numpy as np
from PIL import Image

from common import Geom, raster_city

DATA = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--px", type=int, default=1500, help="longest side, in pixels")
    ap.add_argument("--ink", type=int, default=200, help="street opacity, 0-255")
    ap.add_argument("--force", action="store_true", help="redraw maps that already exist")
    args = ap.parse_args()

    out = DATA / "citymaps"; out.mkdir(parents=True, exist_ok=True)
    cities = sorted(p.stem[5:] for p in DATA.glob("geom_*.npz"))
    if not cities:
        raise SystemExit("no geom_*.npz in data/ — run embed_grid.py first")

    t0 = time.time()
    for i, c in enumerate(cities, 1):
        f = out / f"{c}.png"
        if f.exists() and not args.force:
            print(f"{i}/{len(cities)} {c:<16} skipped (exists)")
            continue
        g = Geom.load(DATA / f"geom_{c}.npz")
        img = raster_city(g.P0, g.P1, g.bb_lo, g.bb_hi, args.px)
        a = (args.ink * np.clip(img, 0, 1)).astype(np.uint8)
        z = np.zeros_like(a)
        Image.fromarray(np.dstack([z, z, z, a]), "RGBA").save(f, optimize=True)
        print(f"{i}/{len(cities)} {c:<16} {img.shape[1]}x{img.shape[0]}")

    size = sum(f.stat().st_size for f in out.glob("*.png")) / 1e6
    print(f"\n{len(list(out.glob('*.png')))} maps in {out}  "
          f"({size:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
