"""Shared geometry, rasterizer and model for the embedding eye test.

Deliberately mirrors the training pipeline. If rendering here drifts from
rendering there, every similarity in the app is measured on the wrong images.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

R, PX, CELL = 500, 112, 500          # crop radius (m), pixels, index cell size (m)


# ---------------------------------------------------------------- geometry ---
class Geom:
    """Street geometry of one city as clippable segments plus a grid index.

    Built from flat arrays, not from a graph, so the app needs neither osmnx
    nor the original .graphml files.
    """

    def __init__(self, P0, P1, cell=CELL):
        self.P0 = np.asarray(P0, np.float64)
        self.P1 = np.asarray(P1, np.float64)

        mid = (self.P0 + self.P1) / 2
        self.max_half = float(np.hypot(*(self.P1 - self.P0).T).max() / 2)
        self.cell, self.lo = cell, mid.min(0)
        self.bb_lo, self.bb_hi = mid.min(0), mid.max(0)

        k = np.floor((mid - self.lo) / cell).astype(np.int64)
        flat = k[:, 0] * 1_000_000 + k[:, 1]
        self.ids = np.argsort(flat, kind="stable")
        uniq, st = np.unique(flat[self.ids], return_index=True)
        self.cells = dict(zip(uniq.tolist(),
                              zip(st.tolist(), np.append(st[1:], len(self.ids)).tolist())))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["P0"], d["P1"])

    def _near(self, c, r):
        """Segment ids that could reach within r of c.

        Segments are filed by midpoint but have length, so the lookup is widened
        by the longest half-segment in the city. Over-fetches; clip() is exact.
        """
        rr = r + self.max_half
        a = np.floor((c - rr - self.lo) / self.cell).astype(np.int64)
        b = np.floor((c + rr - self.lo) / self.cell).astype(np.int64)
        out = []
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                sp = self.cells.get(i * 1_000_000 + j)
                if sp is not None:
                    out.append(self.ids[sp[0]:sp[1]])
        return np.concatenate(out) if out else np.empty(0, np.int64)

    def clip(self, c, r=R):
        """Streets inside the disk, cut exactly at the circle, relative to c."""
        c = np.asarray(c, float)
        ids = self._near(c, r)
        if not len(ids):
            return np.zeros((0, 2, 2))
        p0, p1 = self.P0[ids], self.P1[ids]
        d, f = p1 - p0, p0 - c
        a = (d * d).sum(1); b = 2 * (f * d).sum(1); e = (f * f).sum(1) - r * r
        disc = b * b - 4 * a * e
        ok = (a > 0) & (disc >= 0)
        s = np.sqrt(np.where(ok, disc, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            t0 = np.clip((-b - s) / (2 * a), 0, 1)
            t1 = np.clip((-b + s) / (2 * a), 0, 1)
        k = ok & (t1 > t0)
        return np.stack([p0[k] + t0[k, None] * d[k] - c,
                         p0[k] + t1[k, None] * d[k] - c], axis=1)

    def road_m(self, c, r=R):
        seg = self.clip(c, r)
        return float(np.hypot(*(seg[:, 1] - seg[:, 0]).T).sum()) if len(seg) else 0.0


# ------------------------------------------------------------- rasterizer ---
def raster(seg, px=PX, r=R, width=1.4):
    """Clipped segments -> (px, px) float image, 1 = street, 0 = empty."""
    img = np.zeros((px + 2, px + 2), np.float32)
    if not len(seg):
        return img[1:-1, 1:-1]
    s = px / (2.0 * r)
    a = (seg[:, 0] + r) * s; b = (seg[:, 1] + r) * s
    a[:, 1] = px - a[:, 1]; b[:, 1] = px - b[:, 1]          # y grows downward
    d = b - a
    n = np.maximum(np.ceil(np.hypot(d[:, 0], d[:, 1]) / 0.5).astype(int), 1) + 1
    i = np.repeat(np.arange(len(n)), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    t = (np.arange(n.sum()) - np.repeat(start, n)) / np.maximum(np.repeat(n, n) - 1, 1)
    p = a[i] + t[:, None] * d[i] + 1.0
    x0 = np.floor(p[:, 0]).astype(int); y0 = np.floor(p[:, 1]).astype(int)
    m = (x0 >= 0) & (y0 >= 0) & (x0 < px + 1) & (y0 < px + 1)
    x0, y0 = x0[m], y0[m]
    fx, fy = p[m, 0] - x0, p[m, 1] - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy),       (1, 1, fx * fy)):
        np.add.at(img, (y0 + dy, x0 + dx), w)
    img = np.clip(img, 0, 1)
    img = np.clip(np.maximum.reduce([img, np.roll(img, 1, 0), np.roll(img, -1, 0),
                                     np.roll(img, 1, 1), np.roll(img, -1, 1)])
                  * (width - 1) * 0.7 + img, 0, 1)
    return img[1:-1, 1:-1]


def render(geom, xy, px=PX, r=R):
    """Frozen crop: no jitter, no rotation, no mirror. Evaluation must be deterministic."""
    return raster(geom.clip(xy, r), px, r)


# ------------------------------------------------------------- morphology ---
def metrics(seg):
    """Morphology of one clipped crop, computed from geometry alone.

    orientation_entropy is the headline number: 0 = perfectly aligned grid,
    1 = every bearing equally represented (organic).
    """
    if not len(seg):
        return dict(road_km=0.0, intersections=0, mean_street_m=0.0, orient_entropy=0.0)

    d = seg[:, 1] - seg[:, 0]
    L = np.hypot(d[:, 0], d[:, 1])

    # endpoints shared by 3+ segments are intersections; polyline vertices have degree 2
    pts = np.round(seg.reshape(-1, 2)).astype(np.int64)
    _, cnt = np.unique(pts, axis=0, return_counts=True)
    inter = int((cnt >= 3).sum())

    ang = np.arctan2(d[:, 1], d[:, 0]) % np.pi        # undirected bearing
    h, _ = np.histogram(ang, bins=36, range=(0, np.pi), weights=L)
    p = h / h.sum() if h.sum() > 0 else h
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum() / np.log(36)) if len(nz) else 0.0

    # crossroads vs T-junctions: the classic planned/organic tell
    four, three = int((cnt >= 4).sum()), int((cnt == 3).sum())

    return dict(road_km=float(L.sum() / 1000),
                intersections=inter,
                mean_street_m=float(L.sum() / inter) if inter else 0.0,
                street_len_cv=float(L.std() / L.mean()) if L.mean() > 0 else 0.0,
                four_way_ratio=four / (four + three) if (four + three) else 0.0,
                orient_entropy=H)


# ------------------------------------------------------------------ model ---
class Net(nn.Module):
    """ResNet-18 backbone -> embedding -> projection head.

    Layer names match the training script so checkpoints load unchanged.
    """

    def __init__(self, emb=256, proj=128, in_ch=1):
        super().__init__()
        net = resnet18(weights=None)
        feat = net.fc.in_features
        net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
        self.backbone = net
        self.embed = nn.Sequential(nn.Linear(feat, emb), nn.BatchNorm1d(emb))
        self.project = nn.Sequential(nn.ReLU(), nn.Linear(emb, emb),
                                     nn.ReLU(), nn.Linear(emb, proj))

    def forward(self, x, project=True):
        e = self.embed(self.backbone(x))
        return (e, F.normalize(self.project(e), dim=1)) if project else e


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path, device=None):
    """Rebuild Net from a checkpoint, inferring head sizes from the weights."""
    device = device or pick_device()
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck

    emb = sd["embed.0.weight"].shape[0]
    proj = sd["project.3.weight"].shape[0]
    in_ch = sd["backbone.conv1.weight"].shape[1]

    model = Net(emb=emb, proj=proj, in_ch=in_ch)
    model.load_state_dict(sd)
    model.to(device).eval()

    meta = {k: v for k, v in ck.items() if k != "model"} if isinstance(ck, dict) else {}
    meta.update(emb_dim=emb, proj_dim=proj, in_ch=in_ch, device=device)
    return model, meta


@torch.no_grad()
def embed_images(model, imgs, device=None, batch=128):
    """(N, px, px) float images -> (N, emb) L2-normalised vectors.

    Uses project=False: the embedding is the vector that describes a place.
    The projection head only shapes the training loss and is discarded.
    """
    device = device or next(model.parameters()).device
    out = []
    for i in range(0, len(imgs), batch):
        t = torch.from_numpy(np.asarray(imgs[i:i + batch], np.float32))[:, None].to(device)
        out.append(model(t, project=False).cpu().numpy())
    V = np.concatenate(out)
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-8, None)


# ------------------------------------------------------------- city maps ---
def raster_city(P0, P1, lo, hi, max_px=1500):
    """Whole street network as one greyscale image. Not the training rasterizer —
    this is a backdrop, so it trades antialiasing for covering a whole city."""
    span = np.asarray(hi, float) - np.asarray(lo, float)
    scale = max_px / span.max()
    W, H = int(np.ceil(span[0] * scale)), int(np.ceil(span[1] * scale))
    img = np.zeros((H + 1, W + 1), np.float32)

    a = (np.asarray(P0, float) - lo) * scale
    b = (np.asarray(P1, float) - lo) * scale
    a[:, 1], b[:, 1] = H - a[:, 1], H - b[:, 1]          # y grows downward
    d = b - a

    n = np.maximum(np.ceil(np.hypot(d[:, 0], d[:, 1])).astype(int), 1) + 1
    i = np.repeat(np.arange(len(n)), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    t = (np.arange(n.sum()) - np.repeat(start, n)) / np.maximum(np.repeat(n, n) - 1, 1)
    p = a[i] + t[:, None] * d[i]

    x0 = np.floor(p[:, 0]).astype(int)
    y0 = np.floor(p[:, 1]).astype(int)
    m = (x0 >= 0) & (y0 >= 0) & (x0 <= W) & (y0 <= H)
    np.add.at(img, (y0[m], x0[m]), 1.0)
    return np.clip(img, 0, 1)[:H, :W]
