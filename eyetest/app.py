"""Eye test for street-morphology embeddings.

    streamlit run app.py

Pick a neighbourhood, see what the network thinks looks like it. The real
question the app answers: does a grid query return grids from OTHER cities?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates as image_click

from common import R, PX, Geom, render, metrics, load_model, embed_images

HERE = Path(__file__).parent
DATA = HERE / "data"
CROP_KM2 = np.pi * (R / 1000) ** 2      # area of one crop disk, km^2

# morphology features measured for every crop, in real urbanism terms
FEATURES = ["road_density", "intersection_density", "mean_street_m",
            "street_len_cv", "four_way_ratio", "orient_entropy"]

FEATURE_HELP = {
    "road_density":         "km of street per km² — how much fabric is packed in",
    "intersection_density": "junctions per km² — the standard walkability proxy",
    "mean_street_m":        "metres of street per junction — block size",
    "street_len_cv":        "spread of segment lengths ÷ mean — 0 is metronomic, high is ad hoc",
    "four_way_ratio":       "crossroads ÷ (crossroads + T-junctions) — planned grids run high",
    "orient_entropy":       "0 = every street on one bearing, 1 = all bearings equally used",
}

# (high, low) plain-language readings, used to auto-label clusters
FEATURE_WORDS = {
    "road_density":         ("dense fabric", "sparse fabric"),
    "intersection_density": ("many junctions", "few junctions"),
    "mean_street_m":        ("long blocks", "short blocks"),
    "street_len_cv":        ("irregular lengths", "uniform lengths"),
    "four_way_ratio":       ("crossroads", "T-junctions"),
    "orient_entropy":       ("organic angles", "aligned grid"),
}

st.set_page_config(page_title="Morphology eye test", page_icon="🗺️", layout="wide")


# ------------------------------------------------------------------ loaders ---
@st.cache_resource(show_spinner="loading embeddings…")
def load_embeddings():
    f = DATA / "embeddings.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=False)
    return dict(V=d["V"], city=d["city"].astype(str), xy=d["xy"], thumb=d["thumb"],
                spacing=float(d["spacing"]), min_road=float(d["min_road"]),
                ckpt=str(d["ckpt"]))


@st.cache_resource(show_spinner="loading model…")
def get_model(ckpt):
    return load_model(ckpt)


@st.cache_resource(show_spinner="loading city geometry…")
def get_geom(city):
    f = DATA / f"geom_{city}.npz"
    return Geom.load(f) if f.exists() else None


@st.cache_resource(show_spinner=False)
def get_citymap(city):
    """Pre-rendered street network, drawn once by make_citymaps.py."""
    from PIL import Image
    f = DATA / "citymaps" / f"{city}.png"
    return Image.open(f).convert("RGBA") if f.exists() else None


# ------------------------------------------------------------ projections ---
@st.cache_data(show_spinner="fitting PCA…")
def pca_fit(_V, key, n=50):
    """First n principal coordinates + explained variance.

    _V is underscored so Streamlit does not hash 17 MB on every rerun; `key`
    identifies the embedding set instead.
    """
    Z = _V - _V.mean(0)
    U, S, _s = np.linalg.svd(Z, full_matrices=False)
    n = min(n, len(S))
    var = (S ** 2) / (S ** 2).sum()
    return (U[:, :n] * S[:n]).astype(np.float32), var


@st.cache_data(show_spinner="running t-SNE — this takes a minute…")
def tsne_2d(_V, key, sample, perplexity, seed=0):
    """t-SNE on a subsample, PCA-50 first (the standard recipe — faster and cleaner)."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = (np.arange(len(_V)) if len(_V) <= sample
           else np.sort(rng.choice(len(_V), sample, replace=False)))
    Z = _V[idx] - _V[idx].mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    Z = (U[:, :50] * S[:50]).astype(np.float32)
    Y = TSNE(n_components=2, perplexity=perplexity, init="pca",
             random_state=seed).fit_transform(Z)
    return idx, Y.astype(np.float32)


@st.cache_data(show_spinner="fitting k-means…")
def kmeans_fit(_P, _V, key, k, seed=0):
    """Cluster in the 50-d PCA space — those components hold 99.8% of the
    variance, so it agrees with clustering the raw 256-d vectors (ARI 0.98)
    while fitting 3.5x faster. Centroids are returned in the ORIGINAL space so
    'closest to centroid' stays meaningful."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(_P)
    lab = km.labels_.astype(int)
    cent = np.stack([_V[lab == c].mean(0) for c in range(k)]).astype(np.float32)
    return lab, cent


# ------------------------------------------------------------- morphology ---
def morph_of(geom, c):
    m = metrics(geom.clip(c, R))
    return [m["road_km"] / CROP_KM2, m["intersections"] / CROP_KM2,
            m["mean_street_m"], m["street_len_cv"],
            m["four_way_ratio"], m["orient_entropy"]]


@st.cache_data(show_spinner=False)
def morph_all(_city_arr, _xy, n, cities):
    """Measure every crop. Slow once, then cached to data/morph.npz."""
    f = DATA / "morph.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        if int(d["n"]) == n and all(k in d.files for k in FEATURES):
            return pd.DataFrame({k: d[k] for k in FEATURES})

    out = np.zeros((n, len(FEATURES)), np.float32)
    bar = st.progress(0.0, text="measuring morphology of every crop…")
    for ci, c in enumerate(cities):
        g = get_geom(c)
        if g is not None:
            for i in np.where(_city_arr == c)[0]:
                out[i] = morph_of(g, _xy[i])
        bar.progress((ci + 1) / len(cities), text=f"measuring… {c}")
    bar.empty()
    np.savez_compressed(f, n=n, **{k: out[:, j] for j, k in enumerate(FEATURES)})
    return pd.DataFrame(out, columns=FEATURES)


@st.cache_data(show_spinner=False, max_entries=512)
def render_embed(city, x, y, ckpt):
    """Render one free coordinate and embed it. Cached so that touching an
    unrelated widget does not re-run a model forward pass."""
    g = get_geom(city)
    model, _ = get_model(ckpt)
    img = render(g, (x, y))
    return img, embed_images(model, img[None])[0]


# -------------------------------------------------------------------- utils ---
def pic(img):
    """float 0..1 or uint8 -> black-on-white uint8 for st.image."""
    a = img.astype(np.float32) / 255 if img.dtype == np.uint8 else img
    return (255 * (1 - np.clip(a, 0, 1))).astype(np.uint8)


def neighbours(V, q, k, drop_idx=None, drop_city=None, city=None, min_sim=-1.0):
    s = V @ q
    if drop_idx is not None:
        s[drop_idx] = -np.inf
    if drop_city is not None and city is not None:
        s[city == drop_city] = -np.inf
    idx = np.argsort(-s)[:k]
    keep = s[idx] >= min_sim
    return idx[keep], s[idx][keep]


def show_row(E, idx, sims, cols=6):
    for chunk in range(0, len(idx), cols):
        row = st.columns(cols)
        for c, j in zip(row, range(chunk, min(chunk + cols, len(idx)))):
            i = idx[j]
            with c:
                st.image(pic(E["thumb"][i]), width='stretch')
                st.caption(f"**{E['city'][i]}**  \ncos {sims[j]:.3f}")


def metric_table(E, idx, seed=0):
    """Morphology of a cluster vs a random group of the same size."""
    rng = np.random.default_rng(seed)
    rnd = rng.choice(len(E["V"]), len(idx), replace=False)

    def stats(ii):
        rows = []
        for i in ii:
            g = get_geom(E["city"][i])
            rows.append(metrics(g.clip(E["xy"][i])) if g else {})
        return pd.DataFrame(rows)

    a, b = stats(idx), stats(rnd)
    if a.empty or b.empty:
        return None
    return pd.DataFrame({
        "neighbours (mean)": a.mean(),
        "neighbours (std)": a.std(),
        "random (mean)": b.mean(),
        "random (std)": b.std(),
        "std ratio": a.std() / b.std().replace(0, np.nan),
    }).round(3)


def norm_bounds(g):
    """Usable extent of a city — the box where a full crop still fits."""
    return g.bb_lo + R, g.bb_hi - R


def to_metres(g, xn, yn):
    lo, hi = norm_bounds(g)
    return float(lo[0] + xn * (hi[0] - lo[0])), float(lo[1] + yn * (hi[1] - lo[1]))


def to_norm(g, x, y):
    lo, hi = norm_bounds(g)
    return (float((x - lo[0]) / (hi[0] - lo[0])),
            float((y - lo[1]) / (hi[1] - lo[1])))


def discrete_scale(k, palette=None):
    """Plotly colorscale that renders integer labels 0..k-1 as flat bands."""
    pal = palette or PALETTE
    out = []
    for c in range(k):
        col = pal[c % len(pal)]
        out += [[c / k, col], [(c + 1) / k, col]]
    return out


def _rgb(col):
    """'#2E91E5' or 'rgb(46,145,229)' -> (46, 145, 229)."""
    if col.startswith("#"):
        return tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(int(v) for v in col[4:-1].split(","))


def _alpha(im, f):
    """Scale an RGBA image's alpha channel by f."""
    im = im.copy()
    im.putalpha(im.split()[3].point(lambda v: int(v * f)))
    return im


@st.cache_data(show_spinner=False, max_entries=48)
def city_canvas(city, lab_bytes, k, reg_op, bg_op, show_map, spacing, width):
    """Street map + cluster mosaic composited into one clickable image.

    Returned at `width` px; the caller maps a click's pixel fraction straight
    back to metres, so this must cover exactly bb_lo..bb_hi.
    """
    from PIL import Image
    g = get_geom(city)
    lab = np.frombuffer(lab_bytes, dtype=np.int64)
    idx = np.where(CITY == city)[0]

    bg = get_citymap(city)
    if bg is not None:
        W, H = bg.size
    else:
        span = g.bb_hi - g.bb_lo
        W = 1500; H = max(1, int(round(1500 * span[1] / span[0])))

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    xs, ys, Z = cluster_grid(g, idx, np.zeros(len(CITY), int), XY, spacing)
    Z[:] = np.nan                                   # refill with the passed labels
    if len(xs) and len(ys) and len(idx):
        ix = np.rint((XY[idx, 0] - xs[0]) / spacing).astype(int)
        iy = np.rint((XY[idx, 1] - ys[0]) / spacing).astype(int)
        ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        Z[iy[ok], ix[ok]] = lab[ok]

    if reg_op > 0 and len(xs) and len(ys):
        cells = np.zeros((len(ys), len(xs), 4), np.uint8)
        for c in range(k):
            m = Z == c
            if m.any():
                cells[m] = (*_rgb(PALETTE[c % len(PALETTE)]), 255)
        reg = Image.fromarray(cells[::-1], "RGBA")   # image rows run north->south

        lo, hi = g.bb_lo, g.bb_hi
        sx, sy = hi[0] - lo[0], hi[1] - lo[1]
        half = spacing / 2
        x0 = (xs[0] - half - lo[0]) / sx * W
        x1 = (xs[-1] + half - lo[0]) / sx * W
        y0 = H - (ys[-1] + half - lo[1]) / sy * H
        y1 = H - (ys[0] - half - lo[1]) / sy * H
        box = (max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
        canvas.alpha_composite(_alpha(reg.resize(box, Image.NEAREST), reg_op),
                               (int(round(x0)), int(round(y0))))

    if show_map and bg is not None:
        canvas.alpha_composite(_alpha(bg, bg_op))

    h = max(1, int(round(width * H / W)))
    return canvas.convert("RGB").resize((width, h), Image.LANCZOS)


def draw_marker(img, geom, x, y):
    """Stamp a crosshair at a metre coordinate onto a full-extent city canvas."""
    from PIL import ImageDraw
    shot = img.copy()
    d = ImageDraw.Draw(shot)
    span = geom.bb_hi - geom.bb_lo
    mx = (x - geom.bb_lo[0]) / span[0] * shot.width
    my = shot.height - (y - geom.bb_lo[1]) / span[1] * shot.height
    rad = max(7, shot.width // 90)
    for col, w in (((255, 255, 255), 5), ((220, 0, 0), 3)):
        d.line([(mx - rad, my - rad), (mx + rad, my + rad)], fill=col, width=w)
        d.line([(mx - rad, my + rad), (mx + rad, my - rad)], fill=col, width=w)
    return shot


def click_to_metres(clicked, geom):
    """Component click (pixels) -> position in metres on the city canvas."""
    span = geom.bb_hi - geom.bb_lo
    fx = clicked["x"] / clicked["width"]
    fy = clicked["y"] / clicked["height"]
    return (float(geom.bb_lo[0] + fx * span[0]),
            float(geom.bb_lo[1] + (1 - fy) * span[1]))


def swatch_row(labels, k):
    """Inline cluster legend — heatmaps cannot draw one for themselves."""
    return "".join(
        f"<span style='display:inline-block;margin-right:14px;white-space:nowrap'>"
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"background:{PALETTE[c % len(PALETTE)]};border:1px solid #999;"
        f"vertical-align:middle'></span>&nbsp;{c}</span>"
        for c in range(k) if np.any(labels == c))


def cluster_grid(geom, idx, lab, xy, spacing, r=R):
    """Cluster label per lattice CELL, as a 2-d array.

    embed_grid sampled on a regular lattice, so every point owns one
    spacing x spacing cell. Cells whose point failed the min-road filter were
    never embedded and stay NaN — plotly draws those blank.
    """
    lo, hi = geom.bb_lo + r, geom.bb_hi - r
    xs = np.arange(lo[0], hi[0], spacing)
    ys = np.arange(lo[1], hi[1], spacing)
    Z = np.full((len(ys), len(xs)), np.nan)
    if len(xs) and len(ys) and len(idx):
        ix = np.rint((xy[idx, 0] - xs[0]) / spacing).astype(int)
        iy = np.rint((xy[idx, 1] - ys[0]) / spacing).astype(int)
        ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        Z[iy[ok], ix[ok]] = lab[idx][ok]
    return xs, ys, Z


def label_cluster(z, n=2):
    """Two strongest deviations from average, as words."""
    order = z.abs().sort_values(ascending=False).index[:n]
    return " · ".join(FEATURE_WORDS[f][0 if z[f] > 0 else 1] for f in order)


# --------------------------------------------------------------------- load ---
E = load_embeddings()
if E is None:
    st.title("🗺️ Morphology eye test")
    st.error("No embeddings found.")
    st.markdown("Generate them first:")
    st.code("python embed_grid.py --graphs ../cities --ckpt ../runs/cities57_final.pt")
    st.stop()

V, CITY, XY, N = E["V"], E["city"], E["xy"], len(E["V"])
CKPT_KEY = f"{Path(E['ckpt']).name}:{N}:{V.shape[1]}"
CITIES = sorted(set(CITY.tolist()))
PALETTE = px.colors.qualitative.Dark24

if "q" not in st.session_state:
    st.session_state.q = int(np.random.default_rng(0).integers(N))


# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.header("🗺️ Eye test")
    st.caption(f"**{N:,}** points · **{len(CITIES)}** cities · {V.shape[1]}-d embedding")
    st.caption(f"grid {E['spacing']:.0f} m · min road {E['min_road']/1000:.1f} km")
    st.caption(f"`{Path(E['ckpt']).name}`")

    st.divider()
    k = st.slider("neighbours", 1, 12, 5)
    cross = st.checkbox("cross-city only", value=True,
                        help="Hide neighbours from the query's own city. This is the real "
                             "test — a grid should match grids ELSEWHERE.")
    min_sim = st.slider("min cosine", -1.0, 1.0, -1.0, 0.05)

    st.divider()
    n_clusters = st.slider("k-means clusters", 2, 24, 8,
                           help="Used by the Clusters and City map views.")
    PCS, VAR = pca_fit(V, CKPT_KEY, 50)
    LAB, CENT = kmeans_fit(PCS, V, CKPT_KEY, n_clusters)

    st.divider()
    if st.button("🎲  Random point", width='stretch'):
        st.session_state.q = int(np.random.default_rng().integers(N))

    st.divider()
    st.caption("Reading it: **orient. entropy** near 0 is a rigid grid, "
               "near 1 is organic. **std ratio** under 1 means the cluster is "
               "tighter than chance.")

q = int(st.session_state.q)
qv, qcity = V[q], CITY[q]
nn_idx, nn_sim = neighbours(V, qv, k, drop_idx=q,
                            drop_city=qcity if cross else None, city=CITY, min_sim=min_sim)

# One view at a time. st.tabs would execute all eight bodies on every rerun —
# eight plotly figures, ~5 MB of JSON and a live model inference per click.
VIEWS = ["❓ Guide", "🔍 Neighbours", "📐 Morphology", "🌍 Latent map",
         "🧬 Clusters", "🗺️ City map", "📖 Interpret", "⚖️ Compare"]
view = st.segmented_control("view", VIEWS, default=VIEWS[0],
                            label_visibility="collapsed", key="view")
if view is None:
    view = VIEWS[0]


# --------------------------------------------------------- 1. neighbours ---
if view == "🔍 Neighbours":
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Query")
        st.image(pic(E["thumb"][q]), width='stretch')
        st.markdown(f"**{qcity}** · point `{q}` · cluster `{LAB[q]}`")
        m = metrics(get_geom(qcity).clip(XY[q])) if get_geom(qcity) else None
        if m:
            st.metric("orient. entropy", f"{m['orient_entropy']:.3f}")
            st.caption(f"{m['road_km']:.1f} km road · {m['intersections']} intersections")

    with right:
        st.subheader(f"{len(nn_idx)} nearest" + (" — other cities only" if cross else ""))
        if not len(nn_idx):
            st.warning("Nothing passed the filters. Lower **min cosine**.")
        else:
            show_row(E, nn_idx, nn_sim)
            st.divider()
            hit = pd.Series(CITY[nn_idx]).value_counts()
            c1, c2 = st.columns([2, 1])
            c1.bar_chart(pd.DataFrame({"cos": nn_sim},
                                      index=[f"{CITY[i]}·{i}" for i in nn_idx]))
            c2.write("**cities hit**"); c2.dataframe(hit, width='stretch')


# ---------------------------------------------------------- 2. morphology ---
if view == "📐 Morphology":
    st.caption("If the embedding captures morphology, a neighbour cluster should vary "
               "**less** than a random group of the same size.")
    if not len(nn_idx):
        st.warning("No neighbours to score.")
    else:
        tbl = metric_table(E, np.append(nn_idx, q))
        if tbl is None:
            st.error("Geometry files missing — rerun `embed_grid.py`.")
        else:
            st.dataframe(tbl, width='stretch')
            ratios = tbl["std ratio"].dropna()
            good = (ratios < 1).sum()
            st.metric("metrics tighter than random", f"{good} / {len(ratios)}")
            if good == len(ratios):
                st.success("Every metric is tighter than chance.")
            elif good == 0:
                st.error("Nothing is tighter than chance — the cluster is not morphological.")


# --------------------------------------------------------- 3. latent map ---
if view == "🌍 Latent map":
    st.caption("A 2-d shadow of a 256-d space. PCA keeps global structure but the first "
               "two components rarely carry much variance — try later components, or "
               "t-SNE, which preserves local neighbourhoods instead.")

    comps, var = PCS, VAR
    c1, c2, c3 = st.columns([1.2, 1, 1])
    method = c1.radio("projection", ["PCA", "t-SNE"], horizontal=True)

    if method == "PCA":
        opts = [f"PC{i+1}  ({100*var[i]:.1f}%)" for i in range(12)]
        ix = c2.selectbox("x axis", opts, index=0)
        iy = c3.selectbox("y axis", opts, index=1)
        ax, ay = opts.index(ix), opts.index(iy)
        P, sub = comps[:, [ax, ay]], np.arange(N)
        axis_names = (f"PC{ax+1}", f"PC{ay+1}")
        st.caption(f"These two components hold **{100*(var[ax]+var[ay]):.1f}%** of the "
                   f"variance. First 12 hold {100*var[:12].sum():.1f}%, "
                   f"first 50 hold {100*var[:50].sum():.1f}%.")
    else:
        s1, s2 = c2.slider("sample", 1000, min(12000, N), min(5000, N), 500), \
                 c3.slider("perplexity", 5, 80, 30, 5)
        sub, P = tsne_2d(V, CKPT_KEY, s1, s2)
        axis_names = ("t-SNE 1", "t-SNE 2")
        st.caption(f"Showing **{len(sub):,}** of {N:,} points. t-SNE distances between "
                   "far-apart clusters are not meaningful — only local grouping is.")

    colour_by = st.radio("colour by", ["city", "k-means cluster", "morphology feature"],
                         horizontal=True, key="lm_colour")

    df = pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "city": CITY[sub],
                       "cluster": LAB[sub].astype(str), "i": sub})

    if colour_by == "city":
        fig = px.scatter(df, x="x", y="y", color="city", hover_data=["i", "cluster"],
                         opacity=0.5, height=640, render_mode="webgl")
    elif colour_by == "k-means cluster":
        fig = px.scatter(df.sort_values("cluster", key=lambda s: s.astype(int)),
                         x="x", y="y", color="cluster", hover_data=["i", "city"],
                         opacity=0.6, height=640, render_mode="webgl",
                         color_discrete_sequence=PALETTE)
    else:
        feat = st.selectbox("feature", FEATURES, key="lm_feat",
                            format_func=lambda f: f"{f} — {FEATURE_HELP[f]}")
        M = morph_all(CITY, XY, N, tuple(CITIES))
        df[feat] = M[feat].values[sub]
        fig = px.scatter(df, x="x", y="y", color=feat, hover_data=["i", "city"],
                         opacity=0.6, height=640, render_mode="webgl",
                         color_continuous_scale="Viridis")

    # query and its neighbours, when they survived the subsample
    pos = {g: j for j, g in enumerate(sub)}
    nn_in = [pos[i] for i in nn_idx if i in pos]
    if nn_in:
        fig.add_scatter(x=P[nn_in, 0], y=P[nn_in, 1], mode="markers", name="neighbours",
                        marker=dict(size=13, color="black", symbol="circle-open",
                                    line_width=2))
    if q in pos:
        fig.add_scatter(x=[P[pos[q], 0]], y=[P[pos[q], 1]], mode="markers", name="query",
                        marker=dict(size=18, color="red", symbol="star"))

    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title=axis_names[0], yaxis_title=axis_names[1])
    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------ 4. clusters ---
if view == "🧬 Clusters":
    st.caption(f"k-means, k={n_clusters}, fitted in the top-50 PCA space of the "
               f"{V.shape[1]}-d embeddings — those components carry 99.8% of the "
               "variance, so it matches clustering the raw vectors (ARI 0.98) and "
               "fits 3.5x faster. The question: are these clusters "
               "**morphological types** or just cities?")

    sizes = pd.Series(LAB).value_counts().sort_index()
    M = morph_all(CITY, XY, N, tuple(CITIES))

    # z-scored profile: how each cluster deviates from the global average
    prof = M.groupby(LAB).mean()
    z = (prof - M.mean()) / M.std()
    labels = {c: label_cluster(z.loc[c]) for c in prof.index}

    st.subheader("What each cluster is")
    summary = pd.DataFrame({
        "points": sizes,
        "reading": pd.Series(labels),
        "top city": pd.Series({c: pd.Series(CITY[LAB == c]).value_counts().index[0]
                               for c in prof.index}),
        "top city share": pd.Series({c: pd.Series(CITY[LAB == c]).value_counts(
            normalize=True).iloc[0] for c in prof.index}).round(3),
        "cities present": pd.Series({c: len(set(CITY[LAB == c])) for c in prof.index}),
    })
    st.dataframe(summary, width='stretch')

    share = summary["top city share"].mean()
    if share < 0.25:
        st.success(f"Mean top-city share {share:.1%} — clusters mix cities freely, so they "
                   "are morphological types, not city labels.")
    elif share < 0.5:
        st.info(f"Mean top-city share {share:.1%} — mostly morphological, with some "
                "cities dominating their own cluster.")
    else:
        st.error(f"Mean top-city share {share:.1%} — clusters are largely single cities. "
                 "That is the failure mode: the model learned to recognise places.")

    st.divider()
    st.subheader("Cluster fingerprints")
    st.caption("Standard deviations from the global mean. Red = more than average.")
    fig = px.imshow(z.T, text_auto=".2f", aspect="auto", height=420,
                    color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                    labels=dict(x="cluster", y="", color="z"))
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("What they look like")
    st.caption("The six crops closest to each cluster centroid — the visual definition "
               "of the type.")
    for c in prof.index:
        members = np.where(LAB == c)[0]
        d = np.linalg.norm(V[members] - CENT[c], axis=1)
        pick = members[np.argsort(d)[:6]]
        st.markdown(f"**cluster {c}** · {labels[c]} · {sizes[c]:,} points")
        row = st.columns(6)
        for col, i in zip(row, pick):
            with col:
                st.image(pic(E["thumb"][i]), width='stretch')
                st.caption(f"{CITY[i]}")

    st.divider()
    st.subheader("Which cities land in which cluster")
    comp = pd.crosstab(CITY, LAB, normalize="index")
    fig = px.imshow(comp, aspect="auto", height=900, color_continuous_scale="Blues",
                    labels=dict(x="cluster", y="", color="share of city"))
    st.plotly_chart(fig, width='stretch')
    st.caption("A city spread across many columns has varied fabric. A city that is one "
               "solid block is morphologically monotonous — or the model cannot see "
               "inside it.")


# ------------------------------------------------------------ 5. city map ---
if view == "🗺️ City map":
    st.caption("One city as a mosaic: each grid cell painted by its k-means cluster, "
               "blank where there was too little road to embed. **Click anywhere on "
               "the map to inspect that spot.** Switch to *explore* for pan, zoom and "
               "hover (that mode needs a box-drag to select — Streamlit's plotly build "
               "ignores single clicks). You can also type normalised coordinates: "
               "x and y run 0→1 across the city's usable extent, (0,0) is south-west.")

    cm_city = st.selectbox("city", CITIES, index=CITIES.index(qcity), key="cm_city")
    g = get_geom(cm_city)

    if g is None:
        st.error(f"`data/geom_{cm_city}.npz` missing — rerun `embed_grid.py`.")
    else:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
        xn = c1.number_input("x  (0 = west, 1 = east)", 0.0, 1.0,
                             float(st.session_state.get("xn", 0.5)), 0.005, format="%.3f")
        yn = c2.number_input("y  (0 = south, 1 = north)", 0.0, 1.0,
                             float(st.session_state.get("yn", 0.5)), 0.005, format="%.3f")
        snap = c3.checkbox("snap to nearest grid point", value=True,
                           help="Off renders a fresh crop anywhere; on jumps to an "
                                "already-embedded point so it has a cluster.")
        st.session_state.xn, st.session_state.yn = xn, yn

        if c4.button("🎲  Random point here", width='stretch'):
            sel = np.where(CITY == cm_city)[0]
            gi = int(np.random.default_rng().choice(sel))
            st.session_state.q = gi
            st.session_state.xn, st.session_state.yn = to_norm(g, XY[gi, 0], XY[gi, 1])
            st.rerun()

        x, y = to_metres(g, xn, yn)
        here = np.where(CITY == cm_city)[0]

        gi = None
        if snap and len(here):
            gi = int(here[np.argmin(np.hypot(*(XY[here] - [x, y]).T))])
            x, y = float(XY[gi, 0]), float(XY[gi, 1])

        # ---- display controls
        d1, d2, d3 = st.columns([1, 1.3, 1.3])
        show_map = d1.checkbox("street map", value=True, key="cm_bg")
        bg_op = d2.slider("map strength", 0.0, 1.0, 0.55, 0.05, key="cm_bgop",
                          disabled=not show_map)
        reg_op = d3.slider("region strength", 0.0, 1.0, 0.6, 0.05, key="cm_regop")

        SCALE = discrete_scale(n_clusters)

        mode = st.radio("map mode", ["🖱️ click", "🔎 explore (pan / zoom / hover)"],
                        horizontal=True, key="cm_mode", label_visibility="collapsed")

        if mode.startswith("🖱️"):
            IMW = 820
            canvas = city_canvas(cm_city, LAB[here].astype(np.int64).tobytes(),
                                 n_clusters, reg_op, bg_op, show_map,
                                 float(E["spacing"]), IMW)

            shot = draw_marker(canvas, g, x, y)

            # let the component scale to its container: at a fixed pixel width the
            # image overflows the iframe and the right-hand side is clipped
            clicked_px = image_click(shot, use_column_width="always",
                                     key="cm_click", cursor="crosshair")
            st.markdown(f"<div style='margin-top:6px'><b>cluster</b>&nbsp;&nbsp;"
                        f"{swatch_row(LAB[here], n_clusters)}</div>",
                        unsafe_allow_html=True)

            # Consume each click exactly once, keyed on the component's timestamp.
            # Comparing positions instead would loop forever: with snap on, the
            # stored point never equals the raw click that produced it.
            if clicked_px and clicked_px.get("unix_time") != st.session_state.get(
                    "cm_click_ts"):
                st.session_state.cm_click_ts = clicked_px.get("unix_time")
                cx, cy = click_to_metres(clicked_px, g)
                if snap and len(here):
                    pk = int(here[np.argmin(np.hypot(*(XY[here] - [cx, cy]).T))])
                    st.session_state.q = pk
                    cx, cy = float(XY[pk, 0]), float(XY[pk, 1])
                nx, ny = to_norm(g, cx, cy)
                st.session_state.xn = float(np.clip(nx, 0, 1))
                st.session_state.yn = float(np.clip(ny, 0, 1))
                st.rerun()

        else:
            xs, ys, Z = cluster_grid(g, here, LAB, XY, E["spacing"])

            fig = go.Figure()

            # regions: one flat cell per lattice square, NaN (blank) where unembedded
            fig.add_trace(go.Heatmap(
                x=xs, y=ys, z=Z, colorscale=SCALE, zmin=-0.5, zmax=n_clusters - 0.5,
                showscale=False, hoverongaps=False, opacity=reg_op, xgap=0, ygap=0,
                hovertemplate="cluster %{z:.0f}<extra></extra>", name=""))

            # the selection layer — always invisible, always hit-testable
            fig.add_trace(go.Scatter(
                x=XY[here, 0], y=XY[here, 1], mode="markers", name="points",
                showlegend=False, hovertemplate="cluster %{marker.color:.0f}<extra></extra>",
                marker=dict(size=9,
                            color=LAB[here], colorscale=SCALE,
                            cmin=-0.5, cmax=n_clusters - 0.5,
                            opacity=0.0)))

            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name="here",
                                     hoverinfo="skip", showlegend=False,
                                     marker=dict(size=22, symbol="x", color="red",
                                                 line_width=3)))

            # a legend heatmaps cannot draw for themselves
            for c in range(n_clusters):
                if np.any(LAB[here] == c):
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode="markers", name=str(c),
                        marker=dict(size=11, color=PALETTE[c % len(PALETTE)],
                                    symbol="square")))

            bg = get_citymap(cm_city) if show_map else None
            if bg is not None:
                fig.add_layout_image(dict(source=bg, xref="x", yref="y",
                                          x=float(g.bb_lo[0]), y=float(g.bb_hi[1]),
                                          sizex=float(g.bb_hi[0] - g.bb_lo[0]),
                                          sizey=float(g.bb_hi[1] - g.bb_lo[1]),
                                          sizing="stretch", opacity=bg_op, layer="below"))

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0),
                              clickmode="event+select", dragmode="select",
                              plot_bgcolor="white", xaxis_title="", yaxis_title="",
                              legend=dict(title="cluster", itemsizing="constant"))
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)

            ev = st.plotly_chart(fig, width='stretch', key="citymap",
                                 on_select="rerun", selection_mode=("points", "box", "lasso"))

            pts = []
            try:
                pts = list(ev["selection"]["points"])
            except (TypeError, KeyError):
                pts = []
            if pts:
                # resolve by position: plotly's array encoding changed between versions
                # so customdata is not reliably readable. A box catches several cells,
                # so aim at the middle of the catch.
                sx = float(np.mean([pt["x"] for pt in pts]))
                sy = float(np.mean([pt["y"] for pt in pts]))
                picked = int(here[np.argmin(np.hypot(*(XY[here] - [sx, sy]).T))])
                if picked != st.session_state.get("cm_sel_pt"):
                    st.session_state.cm_sel_pt = picked
                    st.session_state.q = picked
                    st.session_state.xn, st.session_state.yn = to_norm(
                        g, XY[picked, 0], XY[picked, 1])
                    st.rerun()

        # ---- what is actually there
        st.divider()
        a, b = st.columns([1, 3])
        with a:
            if gi is not None:
                img, cl = E["thumb"][gi], LAB[gi]
                st.image(pic(img), width='stretch')
                st.markdown(f"**{cm_city}** · point `{gi}` · cluster `{cl}`")
                fv = V[gi]
            else:
                im, fv = render_embed(cm_city, x, y, E["ckpt"])
                st.image(pic(im), width='stretch')
                st.markdown(f"**{cm_city}** · free point")
            mm = metrics(g.clip((x, y)))
            st.metric("orient. entropy", f"{mm['orient_entropy']:.3f}")
            st.caption(f"{mm['road_km']:.1f} km road · {mm['intersections']} intersections "
                       f"· {100*mm['four_way_ratio']:.0f}% crossroads")
            st.caption(f"x {xn:.3f} · y {yn:.3f}   ({x:,.0f}, {y:,.0f} m)")
        with b:
            st.subheader("Looks like…")
            fi, fs = neighbours(V, fv, k, drop_idx=gi,
                                drop_city=cm_city if cross else None, city=CITY,
                                min_sim=min_sim)
            if len(fi):
                show_row(E, fi, fs)
            else:
                st.warning("Nothing passed the filters.")

        # ---- small multiples: the pattern hunt
        st.divider()
        st.subheader("Same painting, several cities")
        st.caption("Each city normalised to its own 0→1 box, so shapes are stretched — "
                   "you are looking for the **arrangement** of clusters, not the outline. "
                   "Concentric rings mean a centre-to-edge gradient; stripes mean a "
                   "corridor; salt-and-pepper means the model sees no spatial structure.")
        default = [c for c in ["phoenix", "bologna", "kyoto", "manhattan", "cairo", "paris"]
                   if c in CITIES][:6]
        picks = st.multiselect("cities", CITIES, default=default, key="cm_multi")
        if picks:
            ncol = 3
            nrow = int(np.ceil(len(picks) / ncol))
            sub = make_subplots(rows=nrow, cols=ncol, subplot_titles=picks,
                                horizontal_spacing=0.03, vertical_spacing=0.08)
            for n, c in enumerate(picks):
                gg = get_geom(c)
                if gg is None:
                    continue
                sel = np.where(CITY == c)[0]
                gxs, gys, gZ = cluster_grid(gg, sel, LAB, XY, E["spacing"])
                if not gZ.size:
                    continue
                # each city on its own 0..1 box so the mosaics are comparable
                nx_ = (gxs - gxs[0]) / max(gxs[-1] - gxs[0], 1e-9)
                ny_ = (gys - gys[0]) / max(gys[-1] - gys[0], 1e-9)
                sub.add_trace(go.Heatmap(
                    x=nx_, y=ny_, z=gZ, colorscale=SCALE,
                    zmin=-0.5, zmax=n_clusters - 0.5, showscale=False,
                    hoverongaps=False, xgap=0, ygap=0, name=c,
                    hovertemplate=f"{c}<br>cluster %{{z:.0f}}<extra></extra>"),
                    row=n // ncol + 1, col=n % ncol + 1)
            sub.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
            sub.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            sub.update_layout(height=300 * nrow, margin=dict(l=0, r=0, t=30, b=0),
                              plot_bgcolor="white", showlegend=False)
            st.plotly_chart(sub, width='stretch')

            st.caption("Colours match the map above. Blank cells had under "
                       f"{E['min_road']/1000:.1f} km of road inside the crop, so they "
                       "were never embedded — coastline, parkland, desert.")

# ----------------------------------------------------------- 6. interpret ---
if view == "📖 Interpret":
    st.caption("The embedding has no units. This tab gives it some, by tying its axes "
               "and clusters to measurements an urbanist would recognise.")

    M = morph_all(CITY, XY, N, tuple(CITIES))
    comps, var = PCS, VAR
    NPC = 8

    st.subheader("The measurements")
    st.dataframe(pd.DataFrame({"feature": FEATURES,
                               "what it means": [FEATURE_HELP[f] for f in FEATURES],
                               "median": [f"{M[f].median():.2f}" for f in FEATURES],
                               "p5 → p95": [f"{M[f].quantile(.05):.2f} → "
                                            f"{M[f].quantile(.95):.2f}" for f in FEATURES]}
                              ).set_index("feature"), width='stretch')

    st.divider()
    st.subheader("What each PCA axis is measuring")
    C = np.corrcoef(np.c_[comps[:, :NPC], M[FEATURES].values].T)[:NPC, NPC:]
    cdf = pd.DataFrame(C, index=[f"PC{i+1}" for i in range(NPC)], columns=FEATURES)
    fig = px.imshow(cdf, text_auto=".2f", aspect="auto", height=380,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    labels=dict(color="pearson r"))
    st.plotly_chart(fig, width='stretch')

    best = pd.DataFrame({
        "variance": [f"{100*var[i]:.1f}%" for i in range(NPC)],
        "closest feature": [cdf.iloc[i].abs().idxmax() for i in range(NPC)],
        "r": [f"{cdf.iloc[i][cdf.iloc[i].abs().idxmax()]:+.2f}" for i in range(NPC)],
        "reading": [FEATURE_WORDS[cdf.iloc[i].abs().idxmax()][
            0 if cdf.iloc[i][cdf.iloc[i].abs().idxmax()] > 0 else 1] + " at the high end"
            for i in range(NPC)],
    }, index=[f"PC{i+1}" for i in range(NPC)])
    st.dataframe(best, width='stretch')
    st.caption("An |r| above ~0.6 means that axis is close to a restatement of one "
               "measurement. Below ~0.3 it is carrying something these six numbers do "
               "not capture — which is the interesting case, and the reason to keep a "
               "learned embedding at all.")

    st.divider()
    st.subheader("Both ends of an axis")
    pc = st.selectbox("axis", [f"PC{i+1}" for i in range(NPC)], key="int_pc")
    j = int(pc[2:]) - 1
    order = np.argsort(comps[:, j])
    lo_i, hi_i = order[:6], order[-6:][::-1]
    st.markdown(f"**lowest {pc}** — {FEATURE_WORDS[cdf.iloc[j].abs().idxmax()][1]}-ish end")
    row = st.columns(6)
    for col, i in zip(row, lo_i):
        with col:
            st.image(pic(E["thumb"][i]), width='stretch')
            st.caption(f"{CITY[i]}")
    st.markdown(f"**highest {pc}**")
    row = st.columns(6)
    for col, i in zip(row, hi_i):
        with col:
            st.image(pic(E["thumb"][i]), width='stretch')
            st.caption(f"{CITY[i]}")

    st.divider()
    st.subheader("Anchor it to cities you know")
    st.caption("Per-city means. Use this to calibrate the numbers — if you know what "
               "Manhattan feels like, you now know what its intersection density reads as.")
    per_city = M.groupby(CITY).mean().round(2)
    sort_by = st.selectbox("sort by", FEATURES, index=FEATURES.index("orient_entropy"),
                           key="int_sort")
    st.dataframe(per_city.sort_values(sort_by), width='stretch', height=420)

    st.divider()
    st.subheader("Cluster archetypes")
    prof = M.groupby(LAB).mean()
    z = (prof - M.mean()) / M.std()
    arch = pd.DataFrame({
        "reading": pd.Series({c: label_cluster(z.loc[c], 3) for c in prof.index}),
        **{f: prof[f].round(2) for f in FEATURES},
    })
    st.dataframe(arch, width='stretch')
    st.caption("Read a row against the medians above. A cluster with high "
               "intersection_density, low mean_street_m and low orient_entropy is a "
               "tight planned grid; high orient_entropy with low four_way_ratio is "
               "organic medieval fabric; low road_density with long blocks is sprawl.")


# ------------------------------------------------------------- 7. compare ---
if view == "⚖️ Compare":
    st.caption("Two places, one number. Click a spot on either map — does the model "
               "really think Barcelona and Manhattan look alike? Both points snap to "
               "the nearest embedded grid cell, so this compares stored vectors.")

    d1, d2, d3 = st.columns([1, 1.3, 1.3])
    cmp_map = d1.checkbox("street map", value=True, key="cmp_bg")
    cmp_bgop = d2.slider("map strength", 0.0, 1.0, 0.5, 0.05, key="cmp_bgop",
                         disabled=not cmp_map)
    cmp_regop = d3.slider("region strength", 0.0, 1.0, 0.6, 0.05, key="cmp_regop")

    def compare_panel(tag, col, default_city):
        """One side: city picker, clickable mosaic, and the crop it resolves to."""
        with col:
            city = st.selectbox(f"city {tag}", CITIES,
                                index=CITIES.index(default_city), key=f"cmp{tag}_city")
            gg = get_geom(city)
            if gg is None:
                st.error(f"`data/geom_{city}.npz` missing.")
                return None
            idx = np.where(CITY == city)[0]
            xn = float(st.session_state.get(f"cmp{tag}_xn", 0.5))
            yn = float(st.session_state.get(f"cmp{tag}_yn", 0.5))
            cx, cy = to_metres(gg, xn, yn)
            gi = int(idx[np.argmin(np.hypot(*(XY[idx] - [cx, cy]).T))])

            canvas = city_canvas(city, LAB[idx].astype(np.int64).tobytes(),
                                 n_clusters, cmp_regop, cmp_bgop, cmp_map,
                                 float(E["spacing"]), 700)
            clicked = image_click(draw_marker(canvas, gg, XY[gi, 0], XY[gi, 1]),
                                  use_column_width="always",
                                  key=f"cmp{tag}_click", cursor="crosshair")
            st.markdown(f"<div style='margin-top:4px;font-size:0.8em'>"
                        f"{swatch_row(LAB[idx], n_clusters)}</div>",
                        unsafe_allow_html=True)

            thumb, info = st.columns([1, 1])
            thumb.image(pic(E["thumb"][gi]), width='stretch')
            mm = metrics(gg.clip(XY[gi]))
            info.markdown(f"**{city}**  \npoint `{gi}` · cluster `{LAB[gi]}`")
            info.caption(f"entropy {mm['orient_entropy']:.3f}  \n"
                         f"{mm['road_km']:.1f} km road  \n"
                         f"{mm['intersections']} intersections  \n"
                         f"{100*mm['four_way_ratio']:.0f}% crossroads")
            return dict(tag=tag, city=city, g=gg, gi=gi, clicked=clicked, m=mm)

    other = next((c for c in ["manhattan", "barcelona", "bologna"] if c in CITIES
                  and c != qcity), CITIES[min(1, len(CITIES) - 1)])
    colA, colB = st.columns(2)
    A = compare_panel("A", colA, qcity)
    B = compare_panel("B", colB, other)

    # consume clicks after BOTH panels are drawn, so one side cannot abort the other
    for side in (A, B):
        if not side or not side["clicked"]:
            continue
        ts = side["clicked"].get("unix_time")
        if ts != st.session_state.get(f"cmp{side['tag']}_ts"):
            st.session_state[f"cmp{side['tag']}_ts"] = ts
            mx, my = click_to_metres(side["clicked"], side["g"])
            nx, ny = to_norm(side["g"], mx, my)
            st.session_state[f"cmp{side['tag']}_xn"] = float(np.clip(nx, 0, 1))
            st.session_state[f"cmp{side['tag']}_yn"] = float(np.clip(ny, 0, 1))
            st.rerun()

    if A and B:
        st.divider()
        sim = float(V[A["gi"]] @ V[B["gi"]])
        same = LAB[A["gi"]] == LAB[B["gi"]]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("cosine similarity", f"{sim:.4f}")
            st.caption(f"cluster `{LAB[A['gi']]}` vs `{LAB[B['gi']]}` — "
                       + ("same type" if same else "different types"))
            if A["city"] == B["city"]:
                st.info("Same city — the interesting comparison is across cities.")
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sim,
                gauge=dict(axis=dict(range=[-1, 1]),
                           bar=dict(color="#444"),
                           steps=[dict(range=[-1, 0.5], color="#eee"),
                                  dict(range=[0.5, 0.8], color="#cfe"),
                                  dict(range=[0.8, 1], color="#9e9")])))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, width='stretch')

        tbl = pd.DataFrame({f"{A['city']} · {A['gi']}": A["m"],
                            f"{B['city']} · {B['gi']}": B["m"]}).round(3)
        tbl["difference"] = (tbl.iloc[:, 0] - tbl.iloc[:, 1]).abs().round(3)
        st.dataframe(tbl, width='stretch')


# ---------------------------------------------------------------- 8. guide ---
if view == "❓ Guide":
    st.subheader("What this is")
    st.markdown(
        "A neural network was shown millions of 1 km squares of street network "
        "and trained to recognise when two crops came from the same "
        "neighbourhood — never told any city names, never given any labels. "
        "It turns each place into a list of "
        f"{V.shape[1]} numbers.\n\n"
        "Nothing checks that those numbers mean anything. That is what this app "
        "is for. **The test: ask for a grid, and see whether grids come back "
        "from cities on the other side of the world.** If the answer is yes, "
        "the model found street morphology on its own."
    )

    st.divider()
    st.subheader("How to use it")
    st.markdown(
        f"Everything centres on one **query point** — currently `{q}` in "
        f"**{qcity}**. Every view answers a question about it.\n\n"
        "1. Pick a point. **🎲 Random point** in the sidebar, or click a spot "
        "on **City map**.\n"
        "2. Look at **Neighbours** — the crops the model thinks are most alike.\n"
        "3. Keep **cross-city only** switched on. It hides matches from the "
        "query's own city, which is the whole test: matching the block next "
        "door proves nothing, matching Barcelona from Manhattan does.\n"
        "4. Move through the other views for the same point."
    )

    st.divider()
    st.subheader("The views")
    for name, what in [
        ("🔍 Neighbours",
         "The query crop and its closest matches by cosine similarity. Start here."),
        ("📐 Morphology",
         "The numbers behind the eye test. Measures road length, intersections, "
         "street length and orientation for the neighbours, and checks whether "
         "they vary less than a random group of the same size. **std ratio** "
         "below 1 means the model grouped something real."),
        ("🌍 Latent map",
         "All points squashed onto two axes. Colour by city, by cluster, or by a "
         "morphology feature. Clean per-city islands would mean the model only "
         "learned to tell cities apart — mixed ones mean it learned shape."),
        ("🧬 Clusters",
         "k-means over the embeddings: each group's typical crops and its "
         "morphology fingerprint. **Mean top-city share** is the number to "
         "watch — low means the clusters are kinds of street pattern, high "
         "means they are just city labels again."),
        ("🗺️ City map",
         "One city painted cell by cell in its cluster colour, blank where there "
         "was too little street to measure. Click anywhere to inspect that spot. "
         "*Explore* mode adds pan, zoom and hover, where selecting needs a "
         "box-drag rather than a click."),
        ("📖 Interpret",
         "Ties the axes and clusters back to things an urbanist would name: "
         "which measurement each axis tracks, both extremes as real crops, and "
         "the clusters given readable archetypes."),
        ("⚖️ Compare",
         "Two city maps side by side. Click a spot on either and both snap to "
         "the nearest measured point, so the similarity between them is honest."),
    ]:
        st.markdown(f"**{name}** — {what}")

    st.divider()
    st.subheader("Reading the numbers")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Orientation entropy** — how many directions the streets run in.\n\n"
            "Near `0` every street is aligned: Manhattan, Chandigarh, Phoenix.\n\n"
            "Near `1` they run every which way: Bologna, Marrakech, Kathmandu.\n\n"
            "It separates a planned grid from an organic old town on its own."
        )
    with c2:
        st.markdown(
            "**Cosine similarity** — how alike two places look to the model.\n\n"
            "`1.00` identical, `0.90+` a strong match, below `0.70` not much.\n\n"
            "**std ratio** — spread of the neighbours against a random group. "
            "Under `1` the model found a real type; around `1` it found noise."
        )

    st.divider()
    st.subheader("Where the data came from")
    st.caption(
        f"{N:,} crops across {len(CITIES)} cities, sampled on a "
        f"{E['spacing']:.0f} m grid and kept only where there was at least "
        f"{E['min_road']/1000:.1f} km of street inside the frame. Each crop is "
        f"a {R:.0f} m-radius disk rendered to {PX}×{PX} "
        f"pixels, then embedded to {V.shape[1]} dimensions by "
        f"`{Path(E['ckpt']).name}`. Crops are frozen — no rotation or mirroring — "
        "so the same place always measures the same way."
    )
