import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import slic
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_risk_maps import risk_level, slide_segment


cls_rgb = {
    0: (225, 45, 45),
    1: (45, 175, 70),
    2: (190, 195, 205),
}


def read_dewarp_h(tif_p: Path) -> np.ndarray:
    with open(tif_p, "rb") as fh:
        raw = fh.read(180000)
    match = re.search(rb"DewarpHMatrix[=>\"]+([^<\"\n]+)", raw)
    if match:
        vals = [float(val) for val in match.group(1).decode().split(",")]
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    return np.array([[1.7162, 0.0, 415.752], [0.0, 1.7162, 309.814], [0.0, 0.0, 1.0]], dtype=np.float64)


def norm_u8(img: np.ndarray) -> np.ndarray:
    pos = img[img > 0]
    if pos.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, [1.0, 99.0])
    return np.clip((img - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)


def align_ms_rgb(rgb_g: np.ndarray, ms_g: np.ndarray, fb_h: np.ndarray) -> np.ndarray:
    r_sm = norm_u8(cv2.resize(rgb_g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    m_sm = norm_u8(cv2.resize(ms_g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    sift = cv2.SIFT_create(nfeatures=3500)
    kp1, d1 = sift.detectAndCompute(m_sm, None)
    kp2, d2 = sift.detectAndCompute(r_sm, None)
    if d1 is None or d2 is None or len(kp1) < 20 or len(kp2) < 20:
        return fb_h
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m1 for m1, m2 in pairs if m1.distance < 0.72 * m2.distance]
    if len(good) < 20:
        return fb_h
    pts_m = np.float32([kp1[m1.queryIdx].pt for m1 in good]) / 0.5
    pts_r = np.float32([kp2[m1.trainIdx].pt for m1 in good]) / 0.25
    h_mat, mask = cv2.findHomography(pts_m, pts_r, cv2.RANSAC, 6.0)
    if h_mat is None or mask is None or int(mask.sum()) < 20:
        return fb_h
    return h_mat


def extract_superpixel_matrix(
    bgr: np.ndarray,
    ndvi: np.ndarray,
    ndre: np.ndarray,
    gndvi: np.ndarray,
    area_scale: float,
    n_seg: int = 1600,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # construye matriz de 22 rasgos por superpixel slic
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    rf, gf, bf = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    exg = (2.0 * gf - rf - bf) / (rf + gf + bf + 1.0)
    sob = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    segs = slic(rgb, n_segments=n_seg, compactness=14.0, sigma=1.0, start_label=0, channel_axis=-1)
    n_sp = int(segs.max()) + 1
    flat = segs.ravel()
    cnts = np.bincount(flat, minlength=n_sp).astype(np.float32)
    safe = np.maximum(cnts, 1.0)

    def sp_mean(arr: np.ndarray) -> np.ndarray:
        return np.bincount(flat, weights=arr.ravel(), minlength=n_sp) / safe

    def sp_std(arr: np.ndarray, mu: np.ndarray) -> np.ndarray:
        m2 = np.bincount(flat, weights=(arr.ravel() ** 2), minlength=n_sp) / safe
        return np.sqrt(np.maximum(m2 - mu ** 2, 0.0))

    nd_m = sp_mean(ndvi)
    nd_s = sp_std(ndvi, nd_m)
    re_m = sp_mean(ndre)
    re_s = sp_std(ndre, re_m)
    gn_m = sp_mean(gndvi)
    ex_m = sp_mean(exg)
    ex_s = sp_std(exg, ex_m)
    hh_m = sp_mean(hsv[:, :, 0])
    hs_m = sp_mean(hsv[:, :, 1])
    hs_s = sp_std(hsv[:, :, 1], hs_m)
    hv_m = sp_mean(hsv[:, :, 2])
    hv_s = sp_std(hsv[:, :, 2], hv_m)
    ll_m = sp_mean(lab[:, :, 0])
    la_m = sp_mean(lab[:, :, 1])
    lb_m = sp_mean(lab[:, :, 2])
    lb_s = sp_std(lab[:, :, 2], lb_m)
    sb_m = sp_mean(sob)
    sb_s = sp_std(sob, sb_m)

    # aproximacion rapida de percentiles p10 y p90
    nd_p10 = np.clip(nd_m - 1.28 * nd_s, -1.0, 1.0)
    nd_p90 = np.clip(nd_m + 1.28 * nd_s, -1.0, 1.0)
    sb_p90 = sb_m + 1.28 * sb_s
    area_log = np.log1p(cnts * area_scale)

    x_mat = np.column_stack(
        [
            nd_m, nd_s, nd_p10, nd_p90,
            re_m, re_s, gn_m,
            ex_m, ex_s,
            hh_m, hs_m, hs_s, hv_m, hv_s,
            ll_m, la_m, lb_m, lb_s,
            sb_m, sb_s, sb_p90,
            area_log,
        ]
    ).astype(np.float32)
    shadow_sp = hv_m < 42.0
    return segs, x_mat, shadow_sp


def make_heat_overlay(rgb_s: np.ndarray, prob_dry: np.ndarray, vmax: float = 0.85) -> np.ndarray:
    rows, cols = rgb_s.shape[:2]
    pr_s = cv2.resize(prob_dry.astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    pr_s = cv2.GaussianBlur(pr_s, (15, 15), 0)
    hu8 = np.clip(pr_s / vmax * 255.0, 0, 255).astype(np.uint8)
    hrgb = cv2.cvtColor(cv2.applyColorMap(hu8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return (0.50 * rgb_s + 0.50 * hrgb).astype(np.uint8)


def make_seg_overlay(rgb_s: np.ndarray, sem: np.ndarray) -> np.ndarray:
    rows, cols = rgb_s.shape[:2]
    sem_s = cv2.resize(sem, (cols, rows), interpolation=cv2.INTER_NEAREST)
    cmask = np.zeros_like(rgb_s)
    for cid, col in cls_rgb.items():
        cmask[sem_s == cid] = col
    has_c = sem_s < 3
    out = rgb_s.copy()
    out[has_c] = (0.58 * rgb_s[has_c] + 0.42 * cmask[has_c]).astype(np.uint8)
    return out


def save_model_heatmap(
    out_path: Path,
    rgb_s: np.ndarray,
    sem: np.ndarray,
    prob_dry: np.ndarray,
    mod_label: str,
    wp: int,
    yr: str,
    pct_dry: float,
    pct_sana: float,
    pct_soil: float,
    vmax: float = 0.85,
) -> np.ndarray:
    lvl, col_hex = risk_level(pct_dry)
    seg_img = make_seg_overlay(rgb_s, sem)
    heat_img = make_heat_overlay(rgb_s, prob_dry, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=160)
    axes[0].imshow(rgb_s)
    axes[0].set_title("Fotografia Aerea Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(seg_img)
    axes[1].set_title(
        f"Segmentacion {mod_label}\nSeca: {pct_dry:.1f}% | Sana: {pct_sana:.1f}% | Suelo: {pct_soil:.1f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(heat_img)
    axes[2].set_title(f"Mapa de Calor ({mod_label}) - RIESGO {lvl}", fontsize=11, fontweight="bold", color=col_hex)
    axes[2].axis("off")

    fig.suptitle(f"{mod_label} | Waypoint {wp:02d} ({yr}) | Combustible seco: {pct_dry:.1f}%", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return heat_img


def save_five_panel(
    out_path: Path,
    rgb_s: np.ndarray,
    heats: Dict[str, np.ndarray],
    pcts: Dict[str, float],
    wp: int,
    yr: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(23.0, 4.4), dpi=200)
    panels = [
        ("Original RGB", rgb_s),
        (f"KNN (MS+Color)\nSeca: {pcts['knn']:.1f}%", heats["knn"]),
        (f"Random Forest (MS+Color)\nSeca: {pcts['random_forest']:.1f}%", heats["random_forest"]),
        (f"SVM RBF (MS+Color)\nSeca: {pcts['svm']:.1f}%", heats["svm"]),
        (f"YOLO11n-seg (Solo RGB)\nSeca: {pcts['yolo']:.1f}%", heats["yolo"]),
    ]
    for ax, (ttl, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Comparativa de Mapas de Calor por Modelo - Waypoint {wp:02d} ({yr})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    qa_csv = root / "data" / "processed" / "qa_alineacion.csv"
    sp_csv = root / "data" / "yolo_dataset" / "resumen_split.csv"
    raw_dir = root / "data" / "raw"
    mb_dir = root / "data" / "processed" / "multibanda"
    heat_root = root / "results" / "MapasDeCalor"
    fig_dir = root / "results" / "figures"
    met_dir = root / "results" / "metrics"

    parser = argparse.ArgumentParser(description="Genera mapas de calor para KNN, RF, SVM y YOLO.")
    parser.add_argument("--waypoints", type=int, nargs="*", default=None)
    parser.add_argument("--n-segments", type=int, default=1600)
    args = parser.parse_args()

    # crea exactamente 4 carpetas dentro de results/MapasDeCalor
    for folder in ["knn", "random_forest", "svm", "yolo"]:
        (heat_root / folder).mkdir(parents=True, exist_ok=True)

    if args.waypoints:
        wps = set(args.waypoints)
    else:
        wps = set()
        with open(sp_csv, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec["split"] == "test":
                    wps.add(int(rec["waypoint_id"]))

    caps = []
    with open(qa_csv, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if int(rec["waypoint_id"]) in wps:
                caps.append(rec)

    knn_clf = joblib.load(root / "models" / "knn" / "knn_multiespectral.joblib")["pipeline"]
    rf_clf = joblib.load(root / "models" / "random_forest" / "rf_multiespectral.joblib")["model"]
    svm_clf = joblib.load(root / "models" / "svm" / "svm_multiespectral.joblib")["pipeline"]
    yolo_mod = YOLO(str(root / "models" / "yolo" / "best.pt"))

    classic_specs = [
        ("knn", "KNN", knn_clf),
        ("random_forest", "Random Forest", rf_clf),
        ("svm", "SVM RBF", svm_clf),
    ]

    rep_rows: List[Dict] = []
    for cap in tqdm(caps, desc="Mapas de calor 4 modelos"):
        wp = int(cap["waypoint_id"])
        yr = cap["year"]
        cid = cap["capture_id"]
        x0, y0, x1, y1 = [int(val) for val in cap["valid_box"].split(",")]

        full_bgr = cv2.imread(str(raw_dir / yr / f"{cid}_D.JPG"), cv2.IMREAD_COLOR)
        h_f, w_f = full_bgr.shape[:2]
        crop_bgr = full_bgr[y0:y1, x0:x1]
        ch, cw = crop_bgr.shape[:2]

        # alinea bandas calibradas al marco rgb
        mb = tifffile.imread(str(mb_dir / yr / f"wp{wp:02d}_{cid}_MS_aligned.tif")).astype(np.float32)
        g_ms, r_ms, re_ms, nir_ms = mb[0], mb[1], mb[2], mb[3]
        eps = 1e-6
        ndvi_ms = np.clip((nir_ms - r_ms) / (nir_ms + r_ms + eps), -1.0, 1.0)
        ndre_ms = np.clip((nir_ms - re_ms) / (nir_ms + re_ms + eps), -1.0, 1.0)
        gndvi_ms = np.clip((nir_ms - g_ms) / (nir_ms + g_ms + eps), -1.0, 1.0)

        g_tif = raw_dir / yr / f"{cid}_MS_G.TIF"
        fb_h = read_dewarp_h(g_tif) if g_tif.exists() else np.eye(3)
        h_mat = align_ms_rgb(full_bgr[:, :, 1].astype(np.float32), g_ms, fb_h)

        ndvi_c = cv2.warpPerspective(ndvi_ms, h_mat, (w_f, h_f), flags=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        ndre_c = cv2.warpPerspective(ndre_ms, h_mat, (w_f, h_f), flags=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        gndvi_c = cv2.warpPerspective(gndvi_ms, h_mat, (w_f, h_f), flags=cv2.INTER_LINEAR)[y0:y1, x0:x1]

        # resolucion de trabajo para superpixeles rapidos
        wk_w = 1200
        wk_h = int(round(ch * (wk_w / cw)))
        area_sc = float((cw / wk_w) * (ch / wk_h))
        bgr_wk = cv2.resize(crop_bgr, (wk_w, wk_h), interpolation=cv2.INTER_AREA)
        nd_wk = cv2.resize(ndvi_c, (wk_w, wk_h), interpolation=cv2.INTER_LINEAR)
        re_wk = cv2.resize(ndre_c, (wk_w, wk_h), interpolation=cv2.INTER_LINEAR)
        gn_wk = cv2.resize(gndvi_c, (wk_w, wk_h), interpolation=cv2.INTER_LINEAR)

        segs, x_mat, sh_sp = extract_superpixel_matrix(bgr_wk, nd_wk, re_wk, gn_wk, area_sc, args.n_segments)

        tw = 900
        th = int(round(ch * (tw / cw)))
        rgb_s = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_AREA)

        heats_cap: Dict[str, np.ndarray] = {}
        pcts_cap: Dict[str, float] = {}
        row_out: Dict = {"waypoint_id": wp, "year": yr, "capture_id": cid}

        # modelos clasicos
        for folder, lbl, clf in classic_specs:
            probs = clf.predict_proba(x_mat)
            preds = np.argmax(probs, axis=1).astype(np.uint8)
            p_dry_sp = probs[:, 0].astype(np.float32)
            preds[sh_sp] = 255
            p_dry_sp[sh_sp] = 0.0

            sem_map = preds[segs]
            prob_map = p_dry_sp[segs]
            tot = sem_map.size
            p_dry = 100.0 * float((sem_map == 0).sum()) / tot
            p_sana = 100.0 * float((sem_map == 1).sum()) / tot
            p_soil = 100.0 * float((sem_map == 2).sum()) / tot

            out_p = heat_root / folder / f"wp{wp:02d}_{yr}_{cid}_mapa_calor.jpg"
            h_img = save_model_heatmap(out_p, rgb_s, sem_map, prob_map, lbl, wp, yr, p_dry, p_sana, p_soil, vmax=0.85)
            heats_cap[folder] = h_img
            pcts_cap[folder] = p_dry
            row_out[f"pct_seca_{folder}"] = round(p_dry, 2)

        # modelo yolo11n-seg
        sem_y, dry_y, n_win = slide_segment(yolo_mod, crop_bgr, 1024, 0.25, 512, 0.22)
        tot_y = sem_y.size
        py_dry = 100.0 * float((sem_y == 0).sum()) / tot_y
        py_sana = 100.0 * float((sem_y == 1).sum()) / tot_y
        py_soil = 100.0 * float((sem_y == 2).sum()) / tot_y
        out_y = heat_root / "yolo" / f"wp{wp:02d}_{yr}_{cid}_mapa_calor.jpg"
        heats_cap["yolo"] = save_model_heatmap(out_y, rgb_s, sem_y, dry_y, "YOLO11n-seg", wp, yr, py_dry, py_sana, py_soil, vmax=0.65)
        pcts_cap["yolo"] = py_dry
        row_out["pct_seca_yolo"] = round(py_dry, 2)
        rep_rows.append(row_out)

        # figura comparativa de los 4 modelos para wp01 y wp04 de 2025
        if yr == "2025" and wp in (1, 4):
            save_five_panel(fig_dir / f"comparativa_mapas_calor_wp{wp:02d}_{yr}.jpg", rgb_s, heats_cap, pcts_cap, wp, yr)

    rep_rows.sort(key=lambda r: (r["waypoint_id"], r["year"]))
    csv_out = met_dir / "reporte_mapas_calor_modelos.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rep_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rep_rows)

    print(f"Mapas generados en {heat_root} (4 carpetas) | CSV: {csv_out}")
    for rec in rep_rows:
        print(
            f"WP {rec['waypoint_id']:02d} ({rec['year']}) -> "
            f"KNN={rec['pct_seca_knn']:5.1f}% | RF={rec['pct_seca_random_forest']:5.1f}% | "
            f"SVM={rec['pct_seca_svm']:5.1f}% | YOLO={rec['pct_seca_yolo']:5.1f}%"
        )


if __name__ == "__main__":
    main()
