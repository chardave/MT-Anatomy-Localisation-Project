"""Fit the MUSICA display parameters of the nanoDRR simulator to real fluoroscopy frames.

Optimisation problem (one case i, views j; poses fixed at the xvr registration):

    sigma_hat = argmin_sigma  mean_j  l_ij(sigma)  +  lambda * ||u(sigma) - u(sigma_0)||^2

    l_ij(sigma) = mean over valid pixels p of
                  Huber_delta( N(M_sigma(D_ij))(p) - N(F_ij)(p) )
                + mu * Huber_delta( grad N(M_sigma(D_ij))(p) - grad N(F_ij)(p) )

  D_ij = S(CT_i, alpha_ij)  nanoDRR line integral at the xvr pose alpha_ij (rendered once, cached)
  M_sigma                    the simulator's display chain: MUSICA -> percentile window -> invert -> gamma
  F_ij                       the real masked frame; valid pixels exclude the overlay mask
  N(.)                       robust normalisation over valid pixels: (x - median) / (1.4826 * MAD)
  u(sigma)                   unconstrained parameters (bounded sigmoid / log-sigmoid maps); the
                             penalty keeps parameters the data cannot pin down near the GUI defaults

Quantum noise: with --n0, one Poisson sample per view is drawn ONCE (fixed seed, as the simulator
does) and held fixed, so the loss stays differentiable in sigma. A pixel loss cannot judge noise
(random noise never matches random noise) and on its own drives fine-scale texture down. So the
objective adds a texture term per view,
    nu * log( noise(M_sigma(D_ij)) / noise(F_ij) )^2,
noise(.) = robust std of the finest pyramid level after N(.). Give several n0 values
(--n0 1e4 3e4 1e5 ...): each gets its own sigma fit, and the n0 with the lowest total data
objective (pixel + texture term) is chosen. The MUSICA function is imported from 8_1_fluoroscopy_sim_nanodrr.py, so the fitted
values mean the same in the GUI.

Outputs (in --output-dir, or --output-dir/n0_<value> per noise level): drr_cache.npz,
fitted_params.json (simulator key names), loss_curve.png and fit_<view>.png
(real | GUI defaults | fitted | residuals); with several n0 values also n0_sweep.json.

Requirements: fluoro environment (torch, nanodrr, nibabel, Pillow, matplotlib, PySide6 for the import).
Example (case 43, from synthetic_fluoro_generation):

    python fit_musica.py ^
        --settings ..\\xvr_project\\outputs\\registration\\MTAL_KCH_43\\template\\non_contrast\\nanodrr_settings.json ^
        --view AP  ..\\xvr_project\\input\\fluoro_43\\fluoro_non_contrast_AP_masked.png  ..\\xvr_project\\input\\fluoro_43\\checks\\fluoro_non_contrast_AP_mask.png ^
        --view LAT ..\\xvr_project\\input\\fluoro_43\\fluoro_non_contrast_LAT_masked.png ..\\xvr_project\\input\\fluoro_43\\checks\\fluoro_non_contrast_LAT_mask.png ^
        --output-dir ..\\synth_fluoro_gen_test\\MTAL_KCH_43\\template\\musica_fit ^
        --n0 1e4 3e4 1e5 3e5 1e6
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_LEVEL_GAIN = (1.0, 1.0, 1.0, 1.0, 0.6, 0.35, 0.2)

# name, GUI default, lower bound, upper bound, log scale
PARAM_SPECS = [
    ("musica_p", 0.5, 0.1, 1.0, False),
    ("contrast_scale", 0.05, 1e-3, 1.0, True),
    ("noise_floor", 0.003, 1e-4, 0.2, True),
    ("max_gain", 10.0, 1.0, 50.0, True),
    ("residual_gain", 0.1, 0.0, 1.0, False),
    ("gamma", 0.7, 0.1, 3.0, True),
]


def load_simulator():
    spec = importlib.util.spec_from_file_location("fluoro_sim", HERE / "8_1_fluoroscopy_sim_nanodrr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- parameters: bounded values <-> unconstrained u ----------
class Params:
    def __init__(self, n_levels, fixed, device):
        specs = list(PARAM_SPECS) + [(f"level_gain_{k}", DEFAULT_LEVEL_GAIN[k], 0.0, 2.0, False)
                                     for k in range(n_levels)]
        self.specs = [s for s in specs if s[0] not in fixed]
        self.fixed = {s[0]: s[1] for s in specs if s[0] in fixed}
        self.u0 = torch.tensor([self._to_u(s, s[1]) for s in self.specs], dtype=torch.float32, device=device)
        self.n_levels = n_levels

    @staticmethod
    def _to_u(spec, value):
        _, _, lo, hi, log = spec
        if log:
            lo, hi, value = np.log(lo), np.log(hi), np.log(value)
        t = np.clip((value - lo) / (hi - lo), 1e-4, 1 - 1e-4)
        return float(np.log(t / (1 - t)))

    def values(self, u):
        out = dict(self.fixed)
        for k, spec in enumerate(self.specs):
            _, _, lo, hi, log = spec
            t = torch.sigmoid(u[k])
            out[spec[0]] = torch.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * t) if log else lo + (hi - lo) * t
        return out

    def to_simulator(self, u):
        v = {k: float(x) for k, x in self.values(u).items()}
        gains = [v.pop(f"level_gain_{k}") for k in range(self.n_levels)]
        v["level_gain"] = gains + list(DEFAULT_LEVEL_GAIN[self.n_levels:])
        return v


# ---------- display chain M_sigma (differentiable copy of process_drr) ----------
def prepare_input(sim, line_integral, n0, seed):
    """The log image MUSICA receives. Independent of sigma, so it is computed once per view."""
    if n0 is None:
        return (line_integral - line_integral.min()).clamp(min=0)   # add_quantum_noise without noise
    return sim.add_quantum_noise(line_integral, n0, seed=seed)


def display(sim, x, v, n_levels, window=(5.0, 99.7), invert=True):
    gains = tuple(v[f"level_gain_{k}"] for k in range(n_levels))
    enh = sim.musica(x, levels=n_levels, p=v["musica_p"], contrast_scale=v["contrast_scale"],
                     noise_floor=v["noise_floor"], max_gain=v["max_gain"], level_gain=gains,
                     residual_gain=v["residual_gain"])
    q = torch.quantile(enh.flatten(), torch.tensor([window[0] / 100, window[1] / 100],
                                                   dtype=enh.dtype, device=enh.device))
    y = ((enh - q[0]) / (q[1] - q[0] + 1e-12)).clamp(0, 1)
    if invert:
        y = 1 - y
    return y.clamp(min=1e-6) ** v["gamma"]


# ---------- loss ----------
def robust_norm(x, mask):
    vals = x[mask]
    med = vals.median()
    mad = (vals - med).abs().median() * 1.4826 + 1e-6
    return (x - med) / mad


def huber_with_gradients(sim_img, real_n, mask, delta, mu):
    s = robust_norm(sim_img, mask)
    intensity = F.huber_loss(s[mask], real_n[mask], delta=delta)
    mx = mask[:, 1:] & mask[:, :-1]
    my = mask[1:, :] & mask[:-1, :]
    gx_s, gx_r = (s[:, 1:] - s[:, :-1])[mx], (real_n[:, 1:] - real_n[:, :-1])[mx]
    gy_s, gy_r = (s[1:, :] - s[:-1, :])[my], (real_n[1:, :] - real_n[:-1, :])[my]
    grad = F.huber_loss(torch.cat([gx_s, gy_s]), torch.cat([gx_r, gy_r]), delta=delta)
    return intensity + mu * grad, intensity, grad


def noise_level(sim, img, mask):
    """Robust std of the finest Laplacian level inside the (eroded) mask, after normalisation."""
    eroded = -F.max_pool2d(-mask[None, None].float(), 5, stride=1, padding=2)[0, 0] > 0.5
    fine = sim.laplacian_pyramid(robust_norm(img, mask)[None, None], 1)[0][0][0, 0]
    vals = fine[eroded]
    return (vals - vals.median()).abs().median() * 1.4826


# ---------- xvr's gmncc, reimplemented for reporting (defaults of xvr.register: beta 0.5, patches 9/11) ----------
def _patch_ncc(a, b, p):
    if p is not None:
        h, w = (a.shape[-2] // p) * p, (a.shape[-1] // p) * p
        a = a[..., :h, :w].unfold(-2, p, p).unfold(-2, p, p).reshape(a.shape[1], -1, p, p).transpose(0, 1)
        b = b[..., :h, :w].unfold(-2, p, p).unfold(-2, p, p).reshape(b.shape[1], -1, p, p).transpose(0, 1)

    def norm(x):
        return (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.var(dim=(-1, -2), keepdim=True, correction=0) + 1e-5).sqrt()

    return float((norm(a) * norm(b)).mean())


def gmncc(real, sim_img):
    a, b = real[None, None].float(), sim_img[None, None].float()
    gx = torch.tensor([[1., 0, -1], [2, 0, -2], [1, 0, -1]], device=a.device)
    kernel = torch.stack([gx, gx.T])[:, None]
    ga, gb = F.conv2d(a, kernel, padding=1), F.conv2d(b, kernel, padding=1)
    mncc = 0.5 * _patch_ncc(a, b, None) + 0.5 * _patch_ncc(a, b, 9)
    return 0.5 * mncc + 0.5 * _patch_ncc(ga, gb, 11)


# ---------- inputs ----------
def volume_centre_ras(path):
    img = nib.load(str(path))
    return (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]


def render_views(sim, settings, view_names, ct_path, device):
    """nanoDRR line integrals at the registered poses. The camera is built with the isocentre of the
    CT the settings were made for, so another CT in the same scanner space renders the same camera."""
    ref_iso = torch.tensor(volume_centre_ras(settings["ct"]), dtype=torch.float32, device=device)
    subject = sim.Subject.from_filepath(str(ct_path)).to(device)
    out = {}
    for name in view_names:
        s = settings["views"][name]
        k_inv, rt_inv, sdd = sim.make_geometry(
            device, ref_iso, s["rot_x"], s["rot_y"], s["rot_z"], s["trans_x"], s["trans_y"], s["trans_z"],
            s["sdd"], int(s["height"]), int(s["width"]), s["pixel_size"])
        line = sim.render_ct_projection(subject, k_inv, rt_inv, sdd, int(s["height"]), int(s["width"]))
        line = line.sum(dim=1)[0].float()
        if s.get("flip_horizontal", False):
            line = torch.flip(line, dims=[-1])
        out[name] = line.cpu().numpy()
    del subject
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_frame(path, mask_path, shape, border=0):
    real = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if real.shape != tuple(shape):
        raise ValueError(f"{path}: frame is {real.shape}, rendered view is {tuple(shape)}")
    valid = np.ones_like(real, dtype=bool)
    if mask_path is not None:
        overlay = np.asarray(Image.open(mask_path).convert("L")) > 127   # 255 = covered overlay
        if overlay.shape != real.shape:
            raise ValueError(f"{mask_path}: mask is {overlay.shape}, frame is {real.shape}")
        valid &= ~overlay
    if border > 0:   # screen-grab frame edge / collimator line, absent from the DRR
        valid[:border], valid[-border:], valid[:, :border], valid[:, -border:] = False, False, False, False
    return real, valid


# ---------- main ----------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", required=True, type=Path, help="nanodrr_settings.json from 8_cameras_to_nanodrr.py")
    p.add_argument("--view", required=True, nargs=3, action="append", metavar=("NAME", "FRAME", "MASK"),
                   help="View name in the settings (AP/LAT), the masked frame PNG and its overlay mask PNG "
                        "(use - for no mask). Repeat per view.")
    p.add_argument("--ct", type=Path, help="CT to render (default: the CT named in --settings). "
                                           "Must be in the same scanner space and cover the field of view.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--border", type=int, default=10, help="Pixels to ignore along each image edge")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=1.0, help="Huber threshold, in robust standard deviations")
    p.add_argument("--mu", type=float, default=0.5, help="Weight of the gradient term")
    p.add_argument("--lam", type=float, default=1e-3, help="Weight of the pull towards the GUI defaults")
    p.add_argument("--nu", type=float, default=0.5,
                   help="Weight of the noise-texture term (log ratio of fine-scale noise, squared); 0 = off")
    p.add_argument("--fix", nargs="*", default=[], help="Parameter names to hold at their defaults")
    p.add_argument("--n0", type=float, nargs="*", default=None,
                   help="Photon counts per pixel for quantum noise; several values = sweep. Omit for no noise.")
    p.add_argument("--noise-seed", type=int, default=0, help="Seed of the fixed noise sample (simulator default 0)")
    p.add_argument("--rerender", action="store_true", help="Ignore drr_cache.npz and render again")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    ct_path = args.ct or Path(settings["ct"])
    views = [(name, Path(frame), None if mask == "-" else Path(mask)) for name, frame, mask in args.view]
    for name, _, _ in views:
        if name not in settings["views"]:
            raise SystemExit(f"View {name} is not in {args.settings} (has {list(settings['views'])})")

    sim = load_simulator()

    # 1. DRRs: render once, then reuse
    cache = args.output_dir / "drr_cache.npz"
    names = [v[0] for v in views]
    drrs = None
    if cache.exists() and not args.rerender:
        data = np.load(cache, allow_pickle=False)
        if str(data["ct"]) == str(ct_path) and all(n in data.files for n in names):
            drrs = {n: data[n] for n in names}
            print(f"Using cached DRRs from {cache}")
    if drrs is None:
        t0 = time.time()
        print(f"Rendering {names} from {ct_path} ...")
        drrs = render_views(sim, settings, names, ct_path, args.device)
        np.savez_compressed(cache, ct=str(ct_path), **drrs)
        print(f"Rendered in {time.time() - t0:.1f} s, cached to {cache}")

    dev = args.device
    data = []
    for name, frame, mask in views:
        line = torch.from_numpy(drrs[name]).to(dev, torch.float32)[None, None]
        real, valid = load_frame(frame, mask, line.shape[-2:], args.border)
        real_t = torch.from_numpy(real).to(dev, torch.float32)
        valid_t = torch.from_numpy(valid).to(dev)
        data.append(dict(name=name, line=line, real=real_t, valid=valid_t,
                         real_n=robust_norm(real_t, valid_t)))
        print(f"{name}: {valid.mean() * 100:.1f}% of pixels used")

    h, w = data[0]["line"].shape[-2:]
    n_levels = min(7, int(np.log2(min(h, w))) - 3)
    params = Params(n_levels, set(args.fix), dev)
    print(f"MUSICA uses {n_levels} pyramid levels at {h} x {w}; fitting {len(params.specs)} parameters")

    n0_values = args.n0 if args.n0 else [None]
    sweep = []
    for n0 in n0_values:
        out_dir = args.output_dir if len(n0_values) == 1 else args.output_dir / f"n0_{n0:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {'no noise' if n0 is None else f'n0 = {n0:g}'} ===")
        sweep.append(fit_once(args, sim, data, params, n_levels, n0, ct_path, views, out_dir))

    if len(sweep) > 1:
        best = min(sweep, key=lambda r: r["objective_mean"])
        summary = {"chosen_n0": best["n0"], "rule": "lowest objective (pixel loss + texture term)",
                   "runs": [{k: r[k] for k in ("n0", "objective_mean", "loss_mean", "noise_ratio_mean", "output_dir")}
                            for r in sweep]}
        (args.output_dir / "n0_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n n0        objective  pixel loss  noise ratio (sim / real)")
        for r in sweep:
            print(f" {r['n0']:<8g}  {r['objective_mean']:.4f}     {r['loss_mean']:.4f}      {r['noise_ratio_mean']:.2f}"
                  f"{'   <- chosen' if r is best else ''}")
        print(f"Saved n0_sweep.json in {args.output_dir}")


def fit_once(args, sim, data, params, n_levels, n0, ct_path, views, out_dir):
    for d in data:
        d["x"] = prepare_input(sim, d["line"], n0, args.noise_seed)
        d["real_noise"] = noise_level(sim, d["real"], d["valid"])

    def texture_term(img, d):
        return args.nu * torch.log(noise_level(sim, img, d["valid"]) / d["real_noise"]) ** 2

    def total_loss(u):
        v = params.values(u)
        per_view = []
        for d in data:
            img = display(sim, d["x"], v, n_levels)[0, 0]
            loss, _, _ = huber_with_gradients(img, d["real_n"], d["valid"], args.delta, args.mu)
            per_view.append(loss + texture_term(img, d))
        data_term = torch.stack(per_view).mean()
        return data_term + args.lam * ((u - params.u0) ** 2).mean(), data_term

    # Adam on the unconstrained parameters
    u = params.u0.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=args.lr)
    history, best = [], (float("inf"), params.u0.clone())
    t0 = time.time()
    for it in range(args.iterations + 1):
        opt.zero_grad()
        loss, data_term = total_loss(u)
        history.append(float(data_term.detach()))
        if float(loss.detach()) < best[0]:
            best = (float(loss.detach()), u.detach().clone())
        if it % 50 == 0:
            print(f"iter {it:4d}  loss {float(loss.detach()):.5f}  data {float(data_term.detach()):.5f}")
        if it < args.iterations:
            loss.backward()
            opt.step()
    print(f"Fitted in {time.time() - t0:.1f} s")
    u_best = best[1]

    # Report: per-view loss terms, gmncc and noise ratio, GUI defaults vs fitted
    report = {"settings": str(args.settings), "ct": str(ct_path), "frames": {n: str(f) for n, f, _ in views},
              "n0": n0, "noise_seed": args.noise_seed,
              "loss": {"delta": args.delta, "mu": args.mu, "lam": args.lam, "nu": args.nu, "border": args.border}, "views": {}}
    images = {}
    with torch.no_grad():
        for label, uu in (("default", params.u0), ("fitted", u_best)):
            v = params.values(uu)
            for d in data:
                img = display(sim, d["x"], v, n_levels)[0, 0]
                loss, inten, grad = huber_with_gradients(img, d["real_n"], d["valid"], args.delta, args.mu)
                texture = float(texture_term(img, d))
                report["views"].setdefault(d["name"], {})[label] = {
                    "objective": float(loss) + texture, "loss": float(loss), "huber_intensity": float(inten),
                    "huber_gradient": float(grad), "texture_term": texture, "gmncc": gmncc(d["real"], img),
                    "noise_ratio": float(noise_level(sim, img, d["valid"]) / d["real_noise"])}
                images[(d["name"], label)] = img.cpu().numpy()
    report["default_params"] = params.to_simulator(params.u0)
    report["fitted_params"] = params.to_simulator(u_best)
    if n0 is not None:
        report["fitted_params"].update(enhance=True, n0=n0, noise_seed=args.noise_seed)
    (out_dir / "fitted_params.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for d in data:
        n = d["name"]
        r = report["views"][n]
        print(f"{n}: loss {r['default']['loss']:.4f} -> {r['fitted']['loss']:.4f} | "
              f"gmncc {r['default']['gmncc']:.3f} -> {r['fitted']['gmncc']:.3f} | "
              f"noise ratio {r['default']['noise_ratio']:.2f} -> {r['fitted']['noise_ratio']:.2f}")
        if r["default"]["gmncc"] < 0:
            print(f"  WARNING: {n} correlates negatively with the frame; check invert / flip")
    print("Fitted parameters:")
    for k, val in report["fitted_params"].items():
        print(f"  {k:15s} {val}")

    # Figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("data loss (mean over views)")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=110)
    plt.close(fig)

    for d in data:
        n = d["name"]
        real = d["real"].cpu().numpy()
        valid = d["valid"].cpu().numpy()
        res = {}
        for label in ("default", "fitted"):
            s = robust_norm(torch.from_numpy(images[(n, label)]), torch.from_numpy(valid)).numpy()
            res[label] = np.where(valid, np.abs(s - d["real_n"].cpu().numpy()), np.nan)
        vmax = np.nanpercentile(res["default"], 99)
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        panels = [(real, "real frame", "gray", None), (images[(n, "default")], "GUI defaults", "gray", None),
                  (images[(n, "fitted")], "fitted", "gray", None),
                  (res["default"], "|residual| defaults", "magma", vmax),
                  (res["fitted"], "|residual| fitted", "magma", vmax)]
        for a, (img, title, cmap, vm) in zip(axes, panels):
            a.imshow(img, cmap=cmap, vmin=0 if vm else None, vmax=vm)
            a.set_title(title)
            a.axis("off")
        noise = "no noise" if n0 is None else f"n0 = {n0:g}"
        fig.suptitle(f"{n} ({noise}): loss {report['views'][n]['default']['loss']:.4f} -> "
                     f"{report['views'][n]['fitted']['loss']:.4f}")
        fig.tight_layout()
        fig.savefig(out_dir / f"fit_{n}.png", dpi=100)
        plt.close(fig)
    print(f"Saved fitted_params.json, loss_curve.png and fit_<view>.png in {out_dir}")

    fitted = [report["views"][d["name"]]["fitted"] for d in data]
    return {"n0": n0, "output_dir": str(out_dir),
            "objective_mean": float(np.mean([f["objective"] for f in fitted])),
            "loss_mean": float(np.mean([f["loss"] for f in fitted])),
            "noise_ratio_mean": float(np.exp(np.mean([np.log(f["noise_ratio"]) for f in fitted])))}


if __name__ == "__main__":
    main()
