#!/usr/bin/env python3
"""Teeth CBCT progress viewer.

    .venv/bin/python -I teeth_viewer.py build    # extract DICOM, align all scans on the lower jaw, export viewer data
    .venv/bin/python -I teeth_viewer.py photos   # (re)align the 2D panoramic X-rays listed in pano_photos.json
    .venv/bin/python -I teeth_viewer.py serve    # serve docs/ on http://<pi>:8765 (docs/ is also the GitHub Pages site)

Anchor: the basal band of the mandible (lower border, below the tooth roots). Braces move the
teeth but not this bone, so after aligning every scan on it the jaw stays put and only teeth move.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import sys
import time
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
DICOM_DIR = WORK / "dicom"
CACHE_DIR = WORK / "cache"
QA_DIR = WORK / "qa"
VIEWER = ROOT / "docs"
DATA = VIEWER / "data"
PHOTOS_FILE = ROOT / "pano_photos.json"   # [{"file": "x.png", "date": "YYYY-MM-DD"}]
PANO_SCALE = 3                             # photos are exported finer than the CT panorama grid

GRID = 0.3                   # mm, voxel size of the exported volumes
HU_LO, HU_HI = -400.0, 3200.0  # grey window baked into the uint8 volumes
BONE_HU = 300.0
TEETH_HU = 2200.0
BASAL_BAND_MM = 8.0          # how much of the mandible above its lower border counts as anchor


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- extract

def extract_dicoms():
    """Pull only CT slices (*.dcm) out of the archives, one folder per scan date."""
    import pydicom

    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    for zpath in sorted(ROOT.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            groups = {}
            for info in zf.infolist():
                if not info.is_dir() and info.filename.lower().endswith(".dcm"):
                    groups.setdefault(os.path.dirname(info.filename), []).append(info)
            for members in groups.values():
                with zf.open(members[0]) as fh:
                    date = str(pydicom.dcmread(fh, stop_before_pixels=True).StudyDate)
                out = DICOM_DIR / f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                if out.is_dir() and len(list(out.glob("*.dcm"))) >= len(members):
                    continue
                out.mkdir(exist_ok=True)
                for m in members:
                    # basename only: archive paths never escape the target folder
                    with zf.open(m) as src, open(out / os.path.basename(m.filename), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                log(f"extracted {len(members)} slices from {zpath.name} -> {out.name}")


# ----------------------------------------------------------------------------- volumes

def load_scan(date):
    """Read one CBCT series and cache it at GRID mm (box average, so no aliasing)."""
    import SimpleITK as sitk

    cache = CACHE_DIR / f"{date}.mha"
    if cache.exists():
        return sitk.ReadImage(str(cache))
    folder = str(DICOM_DIR / date)
    reader = sitk.ImageSeriesReader()
    sid = reader.GetGDCMSeriesIDs(folder)[0]
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(folder, sid))
    reader.SetOutputPixelType(sitk.sitkFloat32)
    img = reader.Execute()
    factor = max(1, round(GRID / img.GetSpacing()[0]))
    img = sitk.BinShrink(img, [factor] * 3)
    del reader
    img = sitk.Clamp(img, sitk.sitkFloat32, -1000.0, 4000.0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(cache), useCompression=True)
    return img


def basal_mask(ref):
    """Lower-border band of the mandible: the bone that braces do not move."""
    import numpy as np
    import SimpleITK as sitk

    bone = sitk.BinaryThreshold(ref, BONE_HU, 1e9, 1, 0)
    bone = sitk.BinaryMorphologicalOpening(bone, [2, 2, 2])
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(bone), sortByObjectSize=True)
    mand = sitk.GetArrayFromImage(cc) == 1          # largest bone piece in a lower-jaw scan = mandible
    sp = ref.GetSpacing()[2]
    has = mand.any(axis=0)
    zb = np.where(has, mand.argmax(axis=0), 10 ** 6)  # lowest bone voxel per column = lower border
    zmin = zb[has].min()
    keep = has & (zb < zmin + round(28.0 / sp))       # ignore columns whose "bottom" is a crown up top
    z = np.arange(mand.shape[0])[:, None, None]
    band = mand & keep[None] & (z >= zb[None]) & (z < zb[None] + round(BASAL_BAND_MM / sp))
    mask = sitk.GetImageFromArray(band.astype(np.uint8))
    mask.CopyInformation(ref)
    return sitk.BinaryDilate(mask, [2, 2, 2])


def masked_ncc(fixed, moving, mask, tx):
    import numpy as np
    import SimpleITK as sitk

    m = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, -1000.0)
    a = sitk.GetArrayViewFromImage(fixed)
    b = sitk.GetArrayViewFromImage(m)
    k = sitk.GetArrayViewFromImage(mask).astype(bool)
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def register(fixed, moving, mask):
    """Rigid 3D alignment of `moving` onto `fixed`, scored only inside the anchor mask."""
    import numpy as np
    import SimpleITK as sitk

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(stats.GetCentroid(1))

    def method():
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(48)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.3, 7)
        r.SetMetricFixedMask(mask)
        r.SetInterpolator(sitk.sitkLinear)
        return r

    # 1) coarse translation search (±12 mm, 3 mm steps) so a different head position can't trap us
    r = method()
    r.SetOptimizerAsExhaustive([0, 0, 0, 4, 4, 4], stepLength=3.0)
    r.SetOptimizerScales([1, 1, 1, 1, 1, 1])
    r.SetShrinkFactorsPerLevel([4])
    r.SetSmoothingSigmasPerLevel([1.0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(tx, inPlace=True)
    r.Execute(fixed, moving)

    # 2) full rigid refinement, coarse to fine
    r = method()
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300,
                                               relaxationFactor=0.6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([1.0, 0.5, 0.0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(tx, inPlace=True)
    r.Execute(fixed, moving)

    p = tx.GetParameters()
    return tx, {"rot_deg": [round(float(np.degrees(v)), 2) for v in p[:3]],
                "shift_mm": [round(float(v), 2) for v in p[3:]]}


# ----------------------------------------------------------------------------- export helpers

def crop_box(ref):
    """Dentition bounding box (+margin) in voxel indices of the reference grid."""
    import numpy as np
    import SimpleITK as sitk

    a = sitk.GetArrayViewFromImage(ref)
    nz = a.shape[0]
    zs, ys, xs = np.nonzero(a[int(nz * 0.35):] > TEETH_HU)
    pad = round(12.0 / GRID)
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(a.shape[2], x1 + pad), min(a.shape[1], y1 + pad)
    return int(x0), int(y0), int(x1), int(y1)


def to_uint8(img):
    import numpy as np
    import SimpleITK as sitk

    a = sitk.GetArrayViewFromImage(img)
    return np.clip((a - HU_LO) * (255.0 / (HU_HI - HU_LO)), 0, 255).round().astype(np.uint8)


def arch_curve(vol):
    """Fit the dental arch in an axial projection; returns points + outward normals for the panorama."""
    import numpy as np

    nz = vol.shape[0]
    thr = (TEETH_HU - HU_LO) * 255.0 / (HU_HI - HU_LO)
    proj = vol[int(nz * 0.35):].max(axis=0)
    ys, xs = np.nonzero(proj > thr)
    xs, ys = xs.astype(float), ys.astype(float)
    keep = np.ones(len(xs), bool)
    for _ in range(4):  # robust quartic fit y(x), dropping stray bright bits
        c = np.polyfit(xs[keep], ys[keep], 4)
        res = np.abs(np.polyval(c, xs) - ys)
        keep = res < max(3 * np.median(res[keep]), 4.0 / GRID)
    xa, xb = np.percentile(xs[keep], [1, 99])
    xa, xb = xa - 4.0 / GRID, xb + 4.0 / GRID
    xx = np.linspace(xa, xb, 4000)
    yy = np.polyval(c, xx)
    seg = np.hypot(np.diff(xx), np.diff(yy))
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.arange(0, s[-1], 1.0)  # one sample per voxel along the arch
    px, py = np.interp(u, s, xx), np.interp(u, s, yy)
    tx, ty = np.gradient(px), np.gradient(py)
    n = np.hypot(tx, ty)
    nx, ny = ty / n, -tx / n
    if ny[len(ny) // 2] > 0:  # make normals point outward (front teeth: towards the lips = -y)
        nx, ny = -nx, -ny
    r = lambda v: [round(float(t), 2) for t in v]
    return {"x": r(px), "y": r(py), "nx": r(nx), "ny": r(ny)}


def save_qa(ref_u8, mov_u8, mask_np, path):
    """Magenta = reference, green = aligned scan; anchor region outlined in the axial strip."""
    import numpy as np
    from PIL import Image

    zb = np.nonzero(mask_np.any(axis=(1, 2)))[0]
    lo = slice(zb.min(), zb.max() + 1) if len(zb) else slice(0, ref_u8.shape[0] // 3)

    def rgb(a, b):
        return np.stack([a, b, a], axis=-1)

    ax = rgb(ref_u8[lo].max(axis=0), mov_u8[lo].max(axis=0))
    cor = rgb(ref_u8.max(axis=1)[::-1], mov_u8.max(axis=1)[::-1])
    mid = ref_u8.shape[2] // 4
    sag = rgb(ref_u8[:, :, mid][::-1], mov_u8[:, :, mid][::-1])
    h = max(ax.shape[0], cor.shape[0], sag.shape[0])
    w = ax.shape[1] + cor.shape[1] + sag.shape[1]
    out = np.zeros((h, w, 3), np.uint8)
    x = 0
    for im in (ax, cor, sag):
        out[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1]
    Image.fromarray(out).save(path)


# ----------------------------------------------------------------------------- build

def build(args):
    import numpy as np
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(os.cpu_count() or 4)
    if args.extract or not any(DICOM_DIR.glob("*/*.dcm")):
        extract_dicoms()
    dates = sorted(p.name for p in DICOM_DIR.iterdir() if p.is_dir() and any(p.glob("*.dcm")))
    if not dates:
        sys.exit("no DICOM scans found")
    log("scans:", ", ".join(dates))

    heights = {}
    for d in dates:
        img = load_scan(d)
        heights[d] = img.GetSize()[2] * img.GetSpacing()[2]
        log(f"{d}: {img.GetSize()} @ {img.GetSpacing()[0]:.2f} mm, height {heights[d]:.0f} mm")
    # reference = earliest of the short (lower-jaw) scans: that field of view is common to all
    hmin = min(heights.values())
    ref_date = args.ref or next(d for d in dates if heights[d] < hmin + 5)
    ref = load_scan(ref_date)
    log("reference:", ref_date)

    mask = basal_mask(ref)
    mask_np = sitk.GetArrayFromImage(mask).astype(bool)
    log(f"anchor mask: {int(mask_np.sum())} voxels ({mask_np.sum() * GRID ** 3 / 1000:.1f} cm³)")

    x0, y0, x1, y1 = crop_box(ref)
    roi = sitk.RegionOfInterest(ref, [x1 - x0, y1 - y0, ref.GetSize()[2]], [x0, y0, 0])
    nx, ny, nz = roi.GetSize()
    log(f"export grid: {nx}x{ny}x{nz} @ {GRID} mm")

    DATA.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    ref_u8 = to_uint8(roi)
    mask_roi = mask_np[:, y0:y1, x0:x1]
    first = dt.date.fromisoformat(dates[0])
    entries = []
    for d in dates:
        mov = load_scan(d)
        if d == ref_date:
            tx, info = sitk.Euler3DTransform(), {"rot_deg": [0, 0, 0], "shift_mm": [0, 0, 0]}
            ncc0 = ncc1 = 1.0
        else:
            t0 = time.time()
            ncc0 = masked_ncc(ref, mov, mask, sitk.Euler3DTransform())
            tx, info = register(ref, mov, mask)
            ncc1 = masked_ncc(ref, mov, mask, tx)
            log(f"{d}: aligned in {time.time() - t0:.0f}s  shift {info['shift_mm']} mm  rot {info['rot_deg']}°  "
                f"anchor match {ncc0:.3f} -> {ncc1:.3f}")
        u8 = to_uint8(sitk.Resample(mov, roi, tx, sitk.sitkLinear, -1000.0))
        with open(DATA / f"{d}.bin.gz", "wb") as fh:
            fh.write(gzip.compress(u8.tobytes(), compresslevel=6))
        save_qa(ref_u8, u8, mask_roi, QA_DIR / f"{d}.png")
        day = dt.date.fromisoformat(d)
        entries.append({
            "id": d, "label": day.strftime("%d.%m.%Y"), "file": f"data/{d}.bin.gz",
            "months": round((day - first).days / 30.44, 1),
            "height_mm": round(heights[d]), "match_before": round(ncc0, 3), "match_after": round(ncc1, 3), **info,
        })

    # sensible starting views
    thr = (TEETH_HU - HU_LO) * 255.0 / (HU_HI - HU_LO)
    prof = (ref_u8 > thr).sum(axis=(1, 2))
    lo_z = int(nz * 0.35)
    crown_z = lo_z + int(np.argmax(prof[lo_z:int(nz * 0.9)]))
    arch = arch_curve(ref_u8)
    zb = np.nonzero(mask_roi.any(axis=(1, 2)))[0]
    meta = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "spacing": GRID, "dims": [nx, ny, nz], "window_hu": [HU_LO, HU_HI],
        "ref": ref_date, "dates": entries, "arch": arch,
        "anchor_z": [int(zb.min()), int(zb.max())] if len(zb) else None,
        "crown_z": crown_z,
        "defaults": {
            "axial": {"pos": crown_z - round(4 / GRID), "thick": round(12 / GRID)},
            "coronal": {"pos": ny // 2, "thick": ny},
            "sagittal": {"pos": nx // 4, "thick": nx // 2},
            "pano": {"pos": 0, "thick": round(14 / GRID)},
        },
    }
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    build_photos(args)
    log(f"done -> {VIEWER}  (QA overlays in {QA_DIR})")


# ----------------------------------------------------------------------------- 2D panoramic X-rays

def render_pano(vol, arch, thick, avg):
    """Same unrolling as the viewer's panorama: rows top = up, one column per voxel along the arch."""
    import numpy as np

    nz, ny, nx = vol.shape
    ax, ay, anx, any_ = (np.asarray(arch[k])[:, None] for k in ("x", "y", "nx", "ny"))
    s = np.arange(thick)[None] - (thick - 1) // 2
    xs, ys = np.rint(ax + anx * s).astype(int), np.rint(ay + any_ * s).astype(int)
    ok = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    samp = vol[:, np.clip(ys, 0, ny - 1), np.clip(xs, 0, nx - 1)].astype(np.float32)
    samp[:, ~ok] = np.nan
    with np.errstate(all="ignore"):
        out = np.nanmean(samp, axis=2) if avg else np.nanmax(samp, axis=2)
    return np.nan_to_num(out)[::-1]


def implant_blob(img, rows, open_iter, min_bottom=None):
    """The implant among the saturated blobs inside the given row band.

    Scored by height x width of its lower part: a screw (CT) or a crown (X-ray) is thick all the
    way down, while a crowned tooth with root fillings thins out into narrow lines. The braces wire
    glues metal parts together in a projection, so thin links are opened first. With `min_bottom`
    (CT, where the screw saturates) only blobs reaching that deep into the bone count, which drops
    plain crowns. Returns the top-centre of the crown (on an X-ray often only the crown saturates)."""
    import numpy as np
    from scipy import ndimage

    metal = ndimage.binary_opening(img >= 250, iterations=open_iter)
    lab, n = ndimage.label(metal)
    if not n:
        return None, None
    h, w = img.shape
    best, best_score = None, 0
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        if ys.start == 0 or xs.start == 0 or ys.stop == h or xs.stop == w:
            continue  # frame borders, text, cut-off crowns
        cy = (ys.start + ys.stop) / 2
        part = lab[sl] == k
        if not rows[0] * h <= cy <= rows[1] * h or part.sum() < 30:
            continue
        if min_bottom is not None and ys.stop < min_bottom:
            continue
        height = ys.stop - ys.start
        lower_width = np.median(part[int(height * 0.6):].sum(axis=1))
        if height * lower_width > best_score:
            best, best_score = k, height * lower_width
    if best is None:
        return None, None
    blob = lab == best
    rows_k, cols_k = np.nonzero(blob)
    top, bottom = rows_k.min(), rows_k.max()
    if min_bottom is not None:
        # CT: a neighbouring crown may touch the implant crown, so find the screw (widest run of
        # columns near the bottom) and take the crown top right above it
        low = blob[bottom - max(2, (bottom - top) // 5):bottom + 1].any(axis=0)
        runs, start = [], None
        for c, v in enumerate(np.append(low, False)):
            if v and start is None:
                start = c
            elif not v and start is not None:
                runs.append((c - start, start, c))
                start = None
        _, a, b = max(runs)
        cx = (a + b - 1) / 2
        band = (cols_k >= cx - 3) & (cols_k <= cx + 3)
        return (float(cx), float(rows_k[band].min())), blob
    crown = cols_k[rows_k < top + max(3, (bottom - top) // 6)]
    return (float(crown.mean()), float(top)), blob


def register_photo(ct_avg, ct_mip, photo, crown_row, log_name):
    """2D affine fit of a panoramic X-ray onto the CT panorama, judged on jaw bone + implant only."""
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage

    cf, implant_f = implant_blob(ct_mip, (0.15, 0.9), open_iter=2, min_bottom=crown_row + round(8 / GRID))
    cm, _ = implant_blob(photo, (0.5, 0.9), open_iter=6)
    log(f"{log_name}: implant crown top CT {tuple(round(v) for v in cf)}, photo {tuple(round(v) for v in cm)}")
    if cf is None or cm is None:
        raise RuntimeError("implant not found in CT panorama or photo")
    mask = np.zeros(ct_avg.shape, bool)
    mask[int(crown_row + round(6 / GRID)):] = True   # below the crowns: roots + bone
    mask &= ct_avg > 8                                 # inside the CT field of view
    mask |= ndimage.binary_dilation(implant_f, iterations=4)
    F = sitk.GetImageFromArray(ct_avg.astype(np.float32))
    M = sitk.SmoothingRecursiveGaussian(sitk.GetImageFromArray(photo.astype(np.float32)), 1.5)
    K = sitk.GetImageFromArray(mask.astype(np.uint8))

    def method():
        r = sitk.ImageRegistrationMethod()
        r.SetMetricAsMattesMutualInformation(32)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(0.5, 3)
        r.SetMetricFixedMask(K)
        r.SetInterpolator(sitk.sitkLinear)
        return r

    def affine(sx, sy):
        t = sitk.AffineTransform(2)
        t.SetCenter(cf)
        t.SetMatrix([sx, 0, 0, sy])
        t.SetTranslation([cm[0] - cf[0], cm[1] - cf[1]])
        return t

    # implants pinned onto each other, search the unknown X-ray magnification (photo px per CT px)
    best = None
    for sx in np.arange(2.0, 5.01, 0.1):
        for sy in np.arange(2.0, 5.01, 0.1):
            r = method()
            r.SetInitialTransform(affine(sx, sy))
            v = r.MetricEvaluate(F, M)
            if best is None or v < best[0]:
                best = (v, sx, sy)
    tx = affine(best[1], best[2])
    r = method()
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-5, numberOfIterations=400,
                                               relaxationFactor=0.6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([2, 1])
    r.SetSmoothingSigmasPerLevel([1.0, 0.0])
    r.SetInitialTransform(tx, inPlace=True)
    r.Execute(F, M)
    mi_affine = r.GetMetricValue()

    # a panoramic X-ray is not a true projection: magnification drifts along the arch. A very coarse
    # B-spline (4 cells across the whole jaw) bends the bone outline into place but is far too stiff
    # to move single teeth, so real tooth movement survives.
    bs = sitk.BSplineTransformInitializer(F, [4, 1], order=3)
    r = method()
    r.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=400,
                           costFunctionConvergenceFactor=1e7)
    r.SetShrinkFactorsPerLevel([2, 1])
    r.SetSmoothingSigmasPerLevel([1.0, 0.0])
    r.SetMovingInitialTransform(tx)
    r.SetInitialTransform(bs, inPlace=True)
    r.Execute(F, M)
    full = sitk.CompositeTransform(2)
    full.AddTransform(tx)
    full.AddTransform(bs)  # applied first
    disp = np.abs(np.asarray(bs.GetParameters())).max()
    log(f"{log_name}: magnification search {best[1]:.1f}x{best[2]:.1f} -> affine {[round(v, 3) for v in tx.GetMatrix()]}"
        f"  MI {best[0]:.3f} -> {mi_affine:.3f} -> bend {r.GetMetricValue():.3f} (max {disp * GRID:.1f} mm)")
    return full, mask


def build_photos(args=None):
    """Warp the panoramic X-rays into the viewer's panorama frame so they line up with the CT."""
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    meta_path = DATA / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["photos"] = []
    cfg = json.loads(PHOTOS_FILE.read_text()) if PHOTOS_FILE.exists() else []
    if cfg:
        nx, ny, nz = meta["dims"]
        last = meta["dates"][-1]
        vol = np.frombuffer(gzip.decompress((VIEWER / last["file"]).read_bytes()), np.uint8).reshape(nz, ny, nx)
        thick = meta["defaults"]["pano"]["thick"]
        ct_avg = render_pano(vol, meta["arch"], thick, avg=True)
        ct_mip = render_pano(vol, meta["arch"], thick, avg=False)
        crown_z = meta.get("crown_z", meta["defaults"]["axial"]["pos"] + round(4 / GRID))
        crown_row = nz - 1 - crown_z
        first = dt.date.fromisoformat(meta["dates"][0]["id"])
        L = ct_avg.shape[1]
        for item in cfg:
            src = ROOT / item["file"]
            if not src.exists():
                log(f"photo {item['file']} not found, skipped")
                continue
            day = dt.date.fromisoformat(item["date"])
            photo = np.asarray(Image.open(src).convert("L"))
            tx, mask = register_photo(ct_avg, ct_mip, photo, crown_row, item["date"])
            ref = sitk.Image(L * PANO_SCALE, nz * PANO_SCALE, sitk.sitkFloat32)
            ref.SetSpacing([1 / PANO_SCALE] * 2)
            ref.SetOrigin([-(PANO_SCALE - 1) / (2 * PANO_SCALE)] * 2)
            warped = sitk.Resample(sitk.GetImageFromArray(photo.astype(np.float32)), ref, tx, sitk.sitkLinear, 0.0)
            w8 = np.clip(sitk.GetArrayViewFromImage(warped), 0, 255).astype(np.uint8)
            name = f"pano_{item['date']}.png"
            Image.fromarray(w8).save(DATA / name, optimize=True)
            orig = f"photo_{item['date']}{src.suffix.lower()}"
            shutil.copyfile(src, DATA / orig)
            # QA: CT panorama magenta, warped photo green, anchor area dimmed outside
            ct3 = np.kron(ct_avg, np.ones((PANO_SCALE, PANO_SCALE)))
            lo, hi = np.percentile(ct3[ct3 > 0], [2, 99.5])
            ct3 = (np.clip((ct3 - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            QA_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.stack([ct3, w8, ct3], -1)).save(QA_DIR / f"photo_{item['date']}.png")
            meta["photos"].append({
                "id": f"photo:{item['date']}", "date": item["date"], "label": day.strftime("%d.%m.%Y"),
                "file": f"data/{name}", "original": f"data/{orig}", "scale": PANO_SCALE,
                "months": round((day - first).days / 30.44, 1),
            })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    log(f"photos: {len(meta['photos'])}")


# ----------------------------------------------------------------------------- serve

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        if self.path.endswith((".html", "/", ".json")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *a):
        pass


def serve(args):
    httpd = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(VIEWER)))
    log(f"serving {VIEWER} on http://{args.host}:{args.port}")
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="extract + align + export viewer data")
    b.add_argument("--ref", help="reference scan date (YYYY-MM-DD); default: earliest lower-jaw scan")
    b.add_argument("--extract", action="store_true", help="re-extract DICOM from the zip archives")
    b.set_defaults(fn=build)
    p = sub.add_parser("photos", help="align the panoramic X-rays from pano_photos.json (after build)")
    p.set_defaults(fn=build_photos)
    s = sub.add_parser("serve", help="serve the viewer")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=serve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
