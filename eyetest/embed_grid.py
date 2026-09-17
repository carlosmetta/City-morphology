"""Offline pass: grid-sample every city, render, embed, save.

    python embed_grid.py --graphs ../data/graphs.pkl --ckpt ../runs/phoenix_best.pt

Writes into ./data:
    geom_<city>.npz   flat segment arrays, so the app needs no osmnx
    embeddings.npz    vectors, city labels, coordinates, thumbnails
"""
import argparse, pickle, sys, time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from common import R, PX, Geom, render, load_model, embed_images


# ------------------------------------------------------------------ input ---
class LazyGraphs(Mapping):
    """{name: G} where each G is read from its own file on lookup.

    A directory of per-city graphs is far bigger than memory once every one is
    unpickled, and with --reuse-geom most are never touched at all. Loading on
    __getitem__ keeps one city resident at a time.
    """

    def __init__(self, files, read):
        self._files, self._read = dict(files), read

    def __getitem__(self, name):
        return self._read(self._files[name])

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)


def load_graphs(src):
    """Accept a pickled {name: G} dict, or a directory of .pkl / .graphml files."""
    src = Path(src)
    if src.is_dir():
        pkls = sorted(src.glob("*.pkl"))
        if pkls:
            return LazyGraphs(((f.stem, f) for f in pkls),
                              lambda f: pickle.loads(f.read_bytes()))
        files = sorted(src.glob("*.graphml"))
        if not files:
            sys.exit(f"no .pkl or .graphml files in {src}")
        import osmnx as ox
        return LazyGraphs(((f.stem, f) for f in files), ox.load_graphml)
    if src.suffix == ".pkl":
        with open(src, "rb") as fh:
            return pickle.load(fh)
    sys.exit(f"don't know how to read {src}")


def flatten(G):
    """Graph -> straight two-point segments in metres."""
    import osmnx as ox
    if not ox.projection.is_projected(G.graph["crs"]):
        G = ox.project_graph(G)
    G = ox.convert.to_undirected(G)          # else two-way streets count twice

    p0, p1 = [], []
    for u, v, d in G.edges(data=True):
        g = d.get("geometry")
        pts = (np.asarray(g.coords, float) if g is not None else
               np.array([[G.nodes[u]["x"], G.nodes[u]["y"]],
                         [G.nodes[v]["x"], G.nodes[v]["y"]]], float))
        if len(pts) > 1:
            p0.append(pts[:-1]); p1.append(pts[1:])
    return np.concatenate(p0).astype(np.float32), np.concatenate(p1).astype(np.float32)


# ------------------------------------------------------------------- grid ---
def grid_points(geom, spacing, min_road, r=R):
    """Regular lattice over the city, keeping only points with real streets."""
    lo, hi = geom.bb_lo + r, geom.bb_hi - r          # keep the disk inside the data
    if np.any(hi <= lo):
        return np.empty((0, 2))
    xs = np.arange(lo[0], hi[0], spacing)
    ys = np.arange(lo[1], hi[1], spacing)
    P = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    if not len(P):
        return np.empty((0, 2))
    return P[np.array([geom.road_m(p, r) >= min_road for p in P], bool)]


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default="../data/graphs.pkl",
                    help="pickled {name: G} dict, or a directory of .pkl / .graphml")
    ap.add_argument("--ckpt", default="../runs/phoenix_best.pt")
    ap.add_argument("--out", default="data")
    ap.add_argument("--spacing", type=float, default=750.0, help="grid spacing in metres")
    ap.add_argument("--min-road", type=float, default=3000.0,
                    help="minimum metres of street inside a crop")
    ap.add_argument("--reuse-geom", action="store_true",
                    help="skip cities that already have a geom_*.npz")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"loading model  {args.ckpt}")
    model, meta = load_model(args.ckpt)
    print(f"  device {meta['device']} | embedding {meta['emb_dim']}-d "
          f"| trained {meta.get('epochs', meta.get('epoch', '?'))} epochs")
    if "val_acc" in meta:
        print(f"  val acc {100 * meta['val_acc']:.1f}%  (val city: {meta.get('val_city', '?')})")

    print(f"\nloading graphs {args.graphs}")
    graphs = load_graphs(args.graphs)
    print(f"  {len(graphs)} cities\n")

    print(f"{'city':<16}{'segments':>10}{'grid':>8}{'kept':>8}{'sec':>8}")
    print("-" * 50)

    V, cities, XY, thumbs, t_all = [], [], [], [], time.time()
    for name in sorted(graphs):
        t0 = time.time()
        gp = out / f"geom_{name}.npz"

        if args.reuse_geom and gp.exists():
            geom = Geom.load(gp)
        else:
            P0, P1 = flatten(graphs[name])
            np.savez_compressed(gp, P0=P0, P1=P1)
            geom = Geom(P0, P1)

        lo, hi = geom.bb_lo + R, geom.bb_hi - R
        n_grid = max(len(np.arange(lo[0], hi[0], args.spacing)), 0) * \
                 max(len(np.arange(lo[1], hi[1], args.spacing)), 0)

        pts = grid_points(geom, args.spacing, args.min_road)
        if not len(pts):
            print(f"{name:<16}{len(geom.P0):>10,}{n_grid:>8}{0:>8}{time.time()-t0:>8.1f}"
                  f"   <- skipped, no points passed min-road")
            continue

        imgs = np.stack([render(geom, p) for p in pts])
        V.append(embed_images(model, imgs))
        cities += [name] * len(pts)
        XY.append(pts)
        thumbs.append((imgs * 255).astype(np.uint8))

        print(f"{name:<16}{len(geom.P0):>10,}{n_grid:>8}{len(pts):>8}{time.time()-t0:>8.1f}")

    if not V:
        sys.exit("\nno points embedded — lower --min-road or --spacing")

    V = np.concatenate(V).astype(np.float32)
    XY = np.concatenate(XY).astype(np.float64)
    thumbs = np.concatenate(thumbs)
    cities = np.array(cities)

    np.savez_compressed(out / "embeddings.npz", V=V, city=cities, xy=XY, thumb=thumbs,
                        spacing=args.spacing, min_road=args.min_road,
                        ckpt=str(args.ckpt), px=PX, R=R)

    print("-" * 50)
    print(f"{'TOTAL':<16}{'':>18}{len(V):>8}{time.time()-t_all:>8.1f}")
    size = (out / "embeddings.npz").stat().st_size / 1e6
    print(f"\nwrote {out/'embeddings.npz'}  ({size:.1f} MB, {V.shape[1]}-d vectors)")
    print(f"      {len(list(out.glob('geom_*.npz')))} geometry files in {out}/")
    print(f"\nnow run:  streamlit run app.py")


if __name__ == "__main__":
    main()
