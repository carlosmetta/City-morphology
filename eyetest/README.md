# Embedding eye test

Interactive check on what the contrastive model actually learned. There are no
labels in this project, so looking at the latent space is the validation.

The question it answers: **does a grid query return grids from other cities?**

## Run it

**Interpreter note.** On this machine `python` does not exist and the default
`python3` is Homebrew's, which has no torch/osmnx. The one with the packages is:

```bash
export PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
```

One offline pass to grid-sample, render and embed every city:

```bash
cd eyetest && $PY embed_grid.py --graphs ../cities --ckpt ../runs/phoenix_final.pt
```

`../cities` is the 57-city set (one pickled graph per file); `../data/graphs.pkl`
is the original 15 bundled into a single dict. Both work.

Then:

```bash
$PY -m streamlit run app.py
```

## Options

| flag | default | what it does |
|---|---|---|
| `--graphs` | `../data/graphs.pkl` | pickled `{name: G}` dict, or a directory of `.pkl` / `.graphml` (one graph per file, read lazily) |
| `--ckpt` | `../runs/phoenix_best.pt` | trained checkpoint |
| `--spacing` | `750` | grid spacing in metres — lower means more points, slower |
| `--min-road` | `3000` | minimum metres of street inside a crop |
| `--reuse-geom` | off | skip cities that already have a `geom_*.npz` |

`--reuse-geom` is the one worth knowing: flattening graphs is the slow part, so
re-embedding with a new checkpoint takes seconds instead of minutes.

```bash
$PY embed_grid.py --ckpt ../runs/phoenix_final.pt --reuse-geom
```

## The views

One view runs at a time (Streamlit would execute every tab body on every rerun,
which cost about 700 ms per click).

**Neighbours** — a query crop and its nearest neighbours by cosine similarity.
Turn on **cross-city only** in the sidebar; that is the actual test. A Manhattan
grid matching a Barcelona grid means something. A Manhattan grid matching the
block next door does not.

**Morphology** — the numeric version of the eye test. Computes road length,
intersection count, mean street length and orientation entropy for the neighbour
cluster, and compares its spread against a random group of the same size.
`std ratio < 1` means the cluster is tighter than chance.

*Orientation entropy* is the metric to watch: 0 is a perfectly aligned grid,
1 is every bearing equally represented. It separates Manhattan from Bologna on
its own.

**Latent map** — a 2-d projection of every embedding. Pick any two of the first
12 principal components (the first two hold only ~35% of the variance, so the
default view is a poor one) or switch to t-SNE. Colour by city, k-means cluster,
or a morphology feature. If the model had only learned to recognise cities you
would see clean per-city islands and nothing else.

**Clusters** — k-means over the embeddings, with each cluster's morphology
fingerprint, its six crops nearest the centroid, and a city x cluster heatmap.
The number to watch is **mean top-city share**: low means the clusters are
morphological types, high means they are just city labels.

**City map** — one city as a mosaic, each 750 m cell painted by its cluster and
left blank where there was too little road to embed. Click anywhere to inspect
that spot. *Explore* mode swaps in a plotly map with pan, zoom and hover, where
selection needs a box-drag (see Notes). Below, the same painting across several
cities at once — that is where spatial patterns show up.

**Interpret** — ties the axes and clusters to measurements an urbanist would
recognise: which feature each PC correlates with, both ends of any axis as real
crops, per-city anchor values, and named cluster archetypes.

**Compare** — two clickable city maps side by side, each with its own city.
Click a spot on either one; both snap to the nearest embedded cell, so the
cosine similarity is measured between stored vectors. Below the maps: the two
crops, their cluster ids, and a morphology table with a difference column.

## Files

```
common.py        geometry, rasterizer, model — mirrors the training pipeline
embed_grid.py    offline grid -> render -> embed
make_citymaps.py pre-renders the street-map backdrops
app.py           the Streamlit UI
requirements.txt note the plotly < 6 pin
data/            generated (gitignored)
  geom_*.npz       flat segment arrays per city
  embeddings.npz   vectors, cities, coordinates, thumbnails
  morph.npz        morphology features per crop (rebuilt by the app if absent)
  citymaps/*.png   street-network backdrops
```

The app reads only `data/`. It needs neither osmnx nor the original graphs, so
you can copy that folder to another machine and browse there.

## Notes

Crops here are **frozen** — no jitter, rotation or mirroring. Those exist to make
training hard; using them in evaluation would add noise to a measurement that
should be deterministic.

Vectors come from `model(x, project=False)` — the embedding, not the 128-d
projection head. The head only shapes the training loss and is discarded
afterwards, so comparing projections would test the wrong thing.

If rendering in `common.py` ever drifts from rendering in the training script,
every similarity in this app is measured on the wrong images. Keep them in sync.

**Pin plotly below 6.** Plotly 6+ serialises arrays as base64 blobs that
Streamlit cannot read, which silently breaks selection on the explore map.

**Streamlit ignores single clicks on plotly charts.** `on_select` fires on
box/lasso select but not on a click — verified across every `selection_mode` and
`dragmode` combination. That is why the City map's click mode draws the map as an
image and reads coordinates with `streamlit-image-coordinates` instead.
