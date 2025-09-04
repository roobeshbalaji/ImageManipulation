from __future__ import annotations
import argparse, pathlib, sys
from typing import Dict, List, Tuple

import numpy as np
import torch, torchvision
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
import torchvision.transforms.functional as TF

# Grad-CAM for CNNs
from gradcam import GradCAM
from gradcam.utils import visualize_cam

# --- ViT heatmaps ---
from PIL import Image
import timm
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM as ViTGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Defaults (commandline arguments can be used to modify values)
ROOT = pathlib.Path(__file__).parent

# Checkpoints (Download links given in Repo)
DEF_VGG16_MODEL     = ROOT / "../models/vgg16_mcn.pth"
DEF_VGGFACE_MODEL   = ROOT / "../models/vgg_face_dag.pth"
DEF_VITFACE_MODEL   = ROOT / "../models/facevit_pretrained_8.pth"

# dataset folder root
DEF_DATA_ROOT       = ROOT / "../data"

# Output folders
DEF_OUT_VGG16       = ROOT / "../results/vgg16_analysis"
DEF_OUT_VGGFACE     = ROOT / "../results/vggface_analysis"
DEF_OUT_VIT         = ROOT / "../results/vit_analysis"
DEF_OUT_VITFACE     = ROOT / "../results/vitface_analysis"


def minmax(v: np.ndarray) -> np.ndarray:
    """min–max normalize a vector to [0,1]. If constant, returns 0's."""
    v = np.asarray(v); lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(v)


def plot_and_csv(outdir: pathlib.Path, metrics: Dict[str, Dict[str, np.ndarray]], title: str):
    """
    saves:
      - percentRatio_Consistency_comparison_BOTHNORMALIZED.png
      - percentRatio_Consistency_comparison_RAW.png
      - metrics_layerwise.csv
    """
    outdir.mkdir(parents=True, exist_ok=True)

    layers = np.arange(1, int(metrics["Thatcher"]["N_LAY"]) + 1)
    colors = {"Thatcher": "tab:blue", "NonSemantic": "tab:green"}

    # Normalized trends
    plt.figure(figsize=(12, 7))
    for ds in ("Thatcher", "NonSemantic"):
        plt.plot(layers, minmax(metrics[ds]["PR_mean"]), "-o", color=colors[ds],
                 label=f"{ds} percentage ratio (norm)", linewidth=2.1)
        plt.plot(layers, minmax(metrics[ds]["CD"]), "--s", color=colors[ds],
                 label=f"{ds} Consistency (norm)", alpha=0.45, linewidth=2)
    plt.xlabel("Layer"); plt.ylabel("Min–Max Normalized"); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "PercentRatio_Consistency_comparison_BOTHNORMALIZED.png", dpi=300); plt.close()

    # Raw dual-axis
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    for ds in ("Thatcher", "NonSemantic"):
        ax1.plot(layers, metrics[ds]["PR_mean"], "-o", color=colors[ds],
                 label=f"{ds} percentage ratio (raw)", linewidth=2)
    ax1.set_xlabel("Layer"); ax1.set_ylabel("((DU/DI - 1) * 100)"); ax1.grid(True)

    ax2 = ax1.twinx()
    for ds in ("Thatcher", "NonSemantic"):
        ax2.plot(layers, metrics[ds]["CD"], "--s", color=colors[ds],
                 label=f"{ds} Consistency (raw)", alpha=0.45, linewidth=2)
    ax2.set_ylabel("Consistency (%)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, loc="upper right"); plt.tight_layout()
    plt.savefig(outdir / "PercentRatio_Consistency_comparison_RAW.png", dpi=300); plt.close()

    # CSV
    rows = []
    for ds in ("Thatcher", "NonSemantic"):
        rows.append(pd.DataFrame({
            "dataset": ds,
            "layer": layers,
            "PR_mean": metrics[ds]["PR_mean"],
            "CD": metrics[ds]["CD"]
        }))
    pd.concat(rows, ignore_index=True).to_csv(outdir / "metrics_layerwise.csv", index=False)

# Layout resolver
def resolve_layout(data_root: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    """
    Build a dict of class-name -> directory path for each dataset (Thatcher, NonSemantic).
    """
    return {
        "Thatcher": {
            "upright_normal":   data_root / "upright_normal",
            "inverted_normal":  data_root / "inverted_normal",
            "upright_thatcher": data_root / "thatcher" / "upright",
            "inverted_thatcher":data_root / "thatcher" / "inverted",
        },
        "NonSemantic": {
            "upright_normal":   data_root / "upright_normal",
            "inverted_normal":  data_root / "inverted_normal",
            "upright_thatcher": data_root / "NSLM" / "upright",
            "inverted_thatcher":data_root / "NSLM" / "inverted",
        }
    }

# Sanity check for index alignement
def list_and_validate(folders_map: Dict[str, pathlib.Path], ds_name: str) -> Dict[str, List[pathlib.Path]]:
    """
    Given a mapping `class_name -> directory`, glob files and ensure all 4 classes have the same count (index alignment).
    """
    folders = {k: sorted(folders_map[k].glob("*")) for k in ("upright_normal","upright_thatcher","inverted_normal","inverted_thatcher")}
    sizes = [len(v) for v in folders.values()]
    if not sizes or len(set(sizes)) != 1:
        raise RuntimeError(f"[{ds_name}] mismatch counts {sizes} under new layout")
    return folders

# (VGG-16 / VGG-Face)

def preprocess_cv(path: pathlib.Path, img_size: int, AVG: torch.Tensor, device: torch.device):
    """Read an image, resize to (img_size,img_size), convert BGR->RGB, subtract AVG."""
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) - AVG
    return x, rgb


def compute_dataset_metrics_cnn_from_lists(
    feats_fn,
    model: torch.nn.Module,
    AVG: torch.Tensor,
    img_size: int,
    ds_name: str,
    folders: Dict[str, List[pathlib.Path]],
    device: torch.device,
    amp: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute DU, DI, percentage ratio, and consistency using pre-resolved file lists.
    """
    N = len(folders["upright_normal"])
    probe_tensor, _ = preprocess_cv(folders["upright_normal"][0], img_size, AVG, device)
    N_LAY = len(feats_fn(model, probe_tensor))

    DU = np.zeros((N, N_LAY), np.float32)
    DI = np.zeros_like(DU)
    PR = np.zeros_like(DU)
    CD = np.zeros(N_LAY, np.float32)

    for i in trange(N, desc=f"{ds_name}", leave=False):
        imgs = [
            preprocess_cv(folders["upright_normal"][i],  img_size, AVG, device)[0],
            preprocess_cv(folders["upright_thatcher"][i],img_size, AVG, device)[0],
            preprocess_cv(folders["inverted_normal"][i], img_size, AVG, device)[0],
            preprocess_cv(folders["inverted_thatcher"][i],img_size, AVG, device)[0],
        ]
        with torch.autocast(device_type=device.type, enabled=amp):
            F = [feats_fn(model, t) for t in imgs]
        for L in range(N_LAY):
            du = torch.linalg.norm(F[0][L] - F[1][L]).item()
            di = torch.linalg.norm(F[2][L] - F[3][L]).item()
            DU[i, L] = du
            DI[i, L] = di
            PR[i, L] = ((du / di) - 1) * 100 if di > 0 else 0.0

    for L in range(N_LAY):
        CD[L] = (DU[:, L] > DI[:, L]).mean() * 100.0

    return {"PR_mean": PR.mean(0), "CD": CD, "N_LAY": N_LAY}


def gradcam_one_cnn(model, target_layer, img_path, out_dir, AVG, img_size, device, tag: str, flip_lr: bool = False):
    """Generate and save a Grad-CAM overlay as '<tag>.png' in out_dir (CNN)."""
    cam = GradCAM(model, target_layer)
    x, rgb = preprocess_cv(img_path, img_size, AVG, device)
    mask, _ = cam(x.unsqueeze(0))
    overlay = visualize_cam(mask, TF.to_tensor(rgb).to(device), alpha=0.6)[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"{tag}.png"
    im = (overlay.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    if flip_lr:
        im = np.ascontiguousarray(im[:, ::-1, :])
    cv2.imwrite(str(op), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

# VGG-16
def load_vgg16(model_path: pathlib.Path, img_size: int, device: torch.device):
    """Load custom VGG-16 and centering mean image (zeros if 'average' missing)."""
    ckpt = torch.load(model_path, map_location="cpu")
    model = torchvision.models.vgg16(weights=None)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.features.load_state_dict(state, strict=False)
    model.eval().to(device)

    if isinstance(ckpt, dict) and "average" in ckpt:
        a = torch.as_tensor(ckpt["average"]).float()
        AVG = a.view(3,1,1).expand(-1,img_size,img_size) if a.numel()==3 \
              else a.view(img_size,img_size,3).permute(2,0,1)
    else:
        AVG = torch.zeros(3, img_size, img_size)
    return model, AVG.to(device), model.features[29]  # conv5_3


@torch.no_grad()
def feats_vgg16(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Flattened activations from all VGG-16 layers (features + classifier)."""
    z = x.unsqueeze(0)
    out: List[torch.Tensor] = []
    for layer in model.features:
        z = layer(z); out.append(z.flatten())
    z = torch.flatten(z, 1)
    for layer in model.classifier:
        z = layer(z); out.append(z.flatten())
    return out


def run_vgg16_pipeline(args, folders_by_ds: Dict[str, Dict[str, List[pathlib.Path]]]) -> None:
    device = torch.device(args.device)
    model, AVG, target_layer = load_vgg16(args.vgg16_model, args.img_size, device)

    outdir = pathlib.Path(args.out_vgg16); outdir.mkdir(parents=True, exist_ok=True)
    saldir = outdir / "saliency"; saldir.mkdir(exist_ok=True)

    metrics = {}
    tag_map = {
        "Thatcher": {
            "upright_normal":   "normal_upright",
            "inverted_normal":  "normal_inverted",
            "upright_thatcher": "thatcher_upright",
            "inverted_thatcher":"thatcher_inverted",
        },
        "NonSemantic": {
            "upright_normal":   "NSLM_upright",   # only using normal for overlay pairing name scheme
            "inverted_normal":  "NSLM_inverted",  # idem
            "upright_thatcher": "NSLM_upright",
            "inverted_thatcher":"NSLM_inverted",
        }
    }

    for name, folders in folders_by_ds.items():
        if not args.skip_cam:
            # choose first image per class (or more if --cam-per-class > 1)
            for cls, tag_base in tag_map[name].items():
                files = folders[cls][:max(0, args.cam_per_class)]
                for i, fp in enumerate(files):
                    tag = tag_base if args.cam_per_class == 1 else f"{tag_base}_{i+1}"
                    try:
                        flip = (name == "NonSemantic" and cls == "inverted_thatcher")
                        gradcam_one_cnn(model, target_layer, fp, saldir, AVG, args.img_size, device, tag, flip_lr=flip)
                    except Exception as e:
                        print(f"[CAM:{name}/{cls}] {fp.name}: {e}")

        metrics[name] = compute_dataset_metrics_cnn_from_lists(
            feats_vgg16, model, AVG, args.img_size, name, folders, device, amp=args.amp
        )

    plot_and_csv(outdir, metrics, title="VGG-16")
    print("✓ VGG-16: Plots →", outdir)
    print("✓ VGG-16: CSV   →", outdir / "metrics_layerwise.csv")
    print("✓ VGG-16: Saliency overlays →", saldir)


# VGG-Face
def load_vggface(model_path: pathlib.Path, img_size: int, device: torch.device):
    """Load VGG-Face (Vgg_face_dag) and centering mean image from model.meta['mean']."""
    from vgg_face_dag import Vgg_face_dag
    model = Vgg_face_dag().eval()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    means = torch.tensor(model.meta['mean']).float().view(3,1,1)
    AVG = means.expand(-1, img_size, img_size).to(device)
    # Find last conv for CAM
    import torch.nn as nn
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return model, AVG, last_conv


class _VGGFaceFeatHook:
    """Collect flattened activations via forward hooks on all submodules."""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.acts: List[torch.Tensor] = []
        self.handles = []
        for layer in model.children():
            if isinstance(layer, torch.nn.Sequential):
                for subl in layer:
                    self.handles.append(subl.register_forward_hook(self._hook))
            else:
                self.handles.append(layer.register_forward_hook(self._hook))

    def _hook(self, module, input, output):
        try:
            self.acts.append(output.flatten())
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.acts.clear()
        self.model(x.unsqueeze(0))
        return self.acts.copy()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def feats_vggface_factory(model: torch.nn.Module):
    hook = _VGGFaceFeatHook(model)

    @torch.no_grad()
    def feats_fn(_model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        return hook(x)

    feats_fn.close = hook.close  
    return feats_fn


def run_vggface_pipeline(args, folders_by_ds: Dict[str, Dict[str, List[pathlib.Path]]]) -> None:
    device = torch.device(args.device)
    model, AVG, target_layer = load_vggface(args.vggface_model, args.img_size, device)
    feats_fn = feats_vggface_factory(model)

    outdir = pathlib.Path(args.out_vggface); outdir.mkdir(parents=True, exist_ok=True)
    saldir = outdir / "saliency"; saldir.mkdir(exist_ok=True)

    metrics = {}
    tag_map = {
        "Thatcher": {
            "upright_normal":   "normal_upright",
            "inverted_normal":  "normal_inverted",
            "upright_thatcher": "thatcher_upright",
            "inverted_thatcher":"thatcher_inverted",
        },
        "NonSemantic": {
            "upright_normal":   "NSLM_upright",
            "inverted_normal":  "NSLM_inverted",
            "upright_thatcher": "NSLM_upright",
            "inverted_thatcher":"NSLM_inverted",
        }
    }

    for name, folders in folders_by_ds.items():
        if not args.skip_cam:
            for cls, tag_base in tag_map[name].items():
                files = folders[cls][:max(0, args.cam_per_class)]
                for i, fp in enumerate(files):
                    tag = tag_base if args.cam_per_class == 1 else f"{tag_base}_{i+1}"
                    try:
                        flip = (name == "NonSemantic" and cls == "inverted_thatcher")
                        gradcam_one_cnn(model, target_layer, fp, saldir, AVG, args.img_size, device, tag, flip_lr=flip)
                    except Exception as e:
                        print(f"[CAM:{name}/{cls}] {fp.name}: {e}")

        metrics[name] = compute_dataset_metrics_cnn_from_lists(
            feats_fn, model, AVG, args.img_size, name, folders, device, amp=args.amp
        )

    plot_and_csv(outdir, metrics, title="VGG-Face")
    print("✓ VGG-Face: Plots →", outdir)
    print("✓ VGG-Face: CSV   →", outdir / "metrics_layerwise.csv")
    print("✓ VGG-Face: Saliency overlays →", saldir)


# ViT
def load_vit(arch: str, device: torch.device):
    """Create a ViT model from timm (pretrained) and its input transform."""
    model = timm.create_model(arch, pretrained=True).to(device)
    model.eval()
    cfg = model.default_cfg
    h, w = cfg.get("input_size", (3, 224, 224))[1], cfg.get("input_size", (3, 224, 224))[2]
    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    return model, transform


class _ViTFeatHook:
    """Collect per-block token embeddings (post-norm1) and return flattened patch features."""
    def __init__(self, model):
        self.model = model
        self.feat_maps: List[Tuple[int, torch.Tensor]] = []
        self.handles = []
        for idx, blk in enumerate(self.model.blocks):
            h = blk.norm1.register_forward_hook(lambda m, inp, out, idx=idx: self.feat_maps.append((idx, out)))
            self.handles.append(h)

    @torch.no_grad()
    def __call__(self, x_bchw: torch.Tensor) -> List[torch.Tensor]:
        self.feat_maps.clear()
        _ = self.model.forward_features(x_bchw)
        self.feat_maps.sort(key=lambda t: t[0])
        out: List[torch.Tensor] = []
        for _, fmap in self.feat_maps:
            fmap = fmap[0, 1:, :].detach()  # drop CLS; [N_patches, C]
            out.append(fmap.flatten())
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def preprocess_vit(path: pathlib.Path, transform: T.Compose, device: torch.device):
    """PIL → BCHW tensor on device, plus RGB float image in [0,1] for overlays."""
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    rgb = np.float32(np.array(img.resize((tensor.shape[-1], tensor.shape[-2]))) / 255.0)
    return tensor, rgb


def gradcam_one_vit(model, target_layer, reshape_transform, img_path, out_dir, transform, device, tag: str, flip_lr: bool = False):
    """Generate and save a grad-cam overlay for ViT as '<tag>.png' in out_dir."""
    cam = ViTGradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    tensor, rgb_img = preprocess_vit(img_path, transform, device)
    targets = [ClassifierOutputTarget(0)]  # mirrors existing behavior
    cam_map = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_img, cam_map, use_rgb=True, colormap=cv2.COLORMAP_JET)
    if flip_lr:
        overlay = np.ascontiguousarray(overlay[:, ::-1, :])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.png"
    Image.fromarray(overlay).save(out_path)


def compute_dataset_metrics_vit_from_lists(
    feats_fn,
    model,
    transform,
    ds_name: str,
    folders: Dict[str, List[pathlib.Path]],
    device: torch.device,
    amp: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute DU, DI, percentage ratio, and consistency using ViT preprocessor (block-wise)."""
    N = len(folders["upright_normal"])
    probe_tensor, _ = preprocess_vit(folders["upright_normal"][0], transform, device)
    N_BLOCKS = len(feats_fn(probe_tensor))

    DU = np.zeros((N, N_BLOCKS)); DI = np.zeros((N, N_BLOCKS)); PR = np.zeros((N, N_BLOCKS))

    for i in trange(N, desc=f"{ds_name}", leave=False):
        un, ut = preprocess_vit(folders["upright_normal"][i],  transform, device)[0], preprocess_vit(folders["upright_thatcher"][i], transform, device)[0]
        inn, it = preprocess_vit(folders["inverted_normal"][i], transform, device)[0], preprocess_vit(folders["inverted_thatcher"][i], transform, device)[0]
        with torch.autocast(device_type=device.type, enabled=amp):
            f_un, f_ut = feats_fn(un), feats_fn(ut)
            f_in, f_it = feats_fn(inn), feats_fn(it)
        for b in range(N_BLOCKS):
            du = np.linalg.norm((f_un[b] - f_ut[b]).cpu().numpy())
            di = np.linalg.norm((f_in[b] - f_it[b]).cpu().numpy())
            DU[i, b], DI[i, b] = du, di
            PR[i, b] = ((du / di) - 1) * 100 if di > 0 else 0.0

    CD = (DU > DI).mean(axis=0) * 100
    return {"PR_mean": PR.mean(axis=0), "CD": CD, "N_LAY": N_BLOCKS}


def run_vit_pipeline(args, folders_by_ds: Dict[str, Dict[str, List[pathlib.Path]]]) -> None:
    device = torch.device(args.device)
    model, transform = load_vit(args.vit_arch, device)

    outdir = pathlib.Path(args.out_vit); outdir.mkdir(parents=True, exist_ok=True)
    saldir = outdir / "saliency"; saldir.mkdir(exist_ok=True)

    hook = _ViTFeatHook(model)
    def feats_fn(x_bchw: torch.Tensor) -> List[torch.Tensor]:
        return hook(x_bchw)

    target_layer = model.blocks[-1].norm1
    def reshape_transform(tensor: torch.Tensor):
        B, N, C = tensor.size()
        num_patches = N - 1  # exclude CLS
        grid = int(np.sqrt(num_patches))
        if grid * grid != num_patches:
            raise RuntimeError(f"Cannot reshape {num_patches} patches into a square grid.")
        feat = tensor[:, 1:, :].reshape(B, grid, grid, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        return feat

    metrics = {}
    tag_map = {
        "Thatcher": {
            "upright_normal":   "normal_upright",
            "inverted_normal":  "normal_inverted",
            "upright_thatcher": "thatcher_upright",
            "inverted_thatcher":"thatcher_inverted",
        },
        "NonSemantic": {
            "upright_normal":   "NSLM_upright",
            "inverted_normal":  "NSLM_inverted",
            "upright_thatcher": "NSLM_upright",
            "inverted_thatcher":"NSLM_inverted",
        }
    }

    for name, folders in folders_by_ds.items():
        if not args.skip_cam:
            for cls, tag_base in tag_map[name].items():
                files = folders[cls][:max(0, args.cam_per_class)]
                for i, fp in enumerate(files):
                    tag = tag_base if args.cam_per_class == 1 else f"{tag_base}_{i+1}"
                    try:
                        flip = (name == "NonSemantic" and cls == "inverted_thatcher")
                        gradcam_one_vit(model, target_layer, reshape_transform, fp, saldir, transform, device, tag, flip_lr=flip)
                    except Exception as e:
                        print(f"[CAM:{name}/{cls}] {fp.name}: {e}")

        metrics[name] = compute_dataset_metrics_vit_from_lists(
            feats_fn, model, transform, name, folders, device, amp=args.amp
        )

    plot_and_csv(outdir, metrics, title=f"ViT ({args.vit_arch})")
    print("✓ ViT: Plots →", outdir)
    print("✓ ViT: CSV   →", outdir / "metrics_layerwise.csv")
    print("✓ ViT: Saliency overlays →", saldir)


# ViT-Face
def load_vitface(model_path: pathlib.Path, device: torch.device):
    """Load ViT-Face model (vit_face.ViT_face) and return in eval mode."""
    from vit_face import ViT_face
    model = ViT_face(
        image_size=112,
        patch_size=8,
        loss_type='CosFace',
        GPU_ID=None,
        num_class=93431,
        dim=512,
        depth=20,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


IM_MEAN_VITFACE = [0.485, 0.456, 0.406]
IM_STD_VITFACE  = [0.229, 0.224, 0.225]


def preprocess_vitface(path: pathlib.Path, size: int, device: torch.device):
    """PIL → BCHW tensor (normalized to ImageNet stats) and PIL image for overlays."""
    img = Image.open(path).convert("RGB").resize((size, size))
    tensor = TF.normalize(T.ToTensor()(img), mean=IM_MEAN_VITFACE, std=IM_STD_VITFACE)\
                .unsqueeze(0).to(device)
    return tensor, img


class _ViTFaceFeatHook:
    """Capture CLS token embedding at each transformer layer during the ORIGINAL forward."""
    def __init__(self, model):
        self.model = model
        self.maps: Dict[str, torch.Tensor] = {}
        self.names: List[str] = []
        self.handles = []

        layers = getattr(getattr(model, "transformer", None), "layers", None)
        if layers is None:
            raise RuntimeError("ViTFace: model.transformer.layers not found.")
        N = len(layers)
        for i in range(N):
            try:
                block = layers[i][0]
            except Exception:
                block = layers[i]
            name = f"layer{i}"
            self.names.append(name)
            self.handles.append(block.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, outp):
            out = outp[0] if isinstance(outp, tuple) else outp
            self.maps[name] = out[:, 0, :].detach()  # CLS
        return hook

    @torch.no_grad()
    def __call__(self, x_bchw: torch.Tensor) -> List[torch.Tensor]:
        self.maps.clear()
        _ = self.model(x_bchw)
        feats: List[torch.Tensor] = []
        for n in self.names:
            v = self.maps[n]
            feats.append(v[0].flatten())
        return feats

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def vitface_single_heatmap(
    model, img_path: pathlib.Path, out_path: pathlib.Path,
    size: int = 112, head_reduce: str = "mean", layer_mode: str = "dynamic", flip_lr: bool = False
):
    """Produce ONE attention-based heatmap for ViT-Face and save to out_path."""
    img_pil = Image.open(img_path).convert('RGB').resize((size, size))
    t = TF.normalize(T.ToTensor()(img_pil), mean=IM_MEAN_VITFACE, std=IM_STD_VITFACE)[None].to(next(model.parameters()).device)
    emb, A = model.forward_with_attention(t)  # list of [1,H,T,T]
    assert len(A) > 0, "No attentions returned by forward_with_attention."

    vecs = []
    for a in A:
        ah = a[0]  # [H,T,T]
        attn = ah.mean(dim=0) if head_reduce == "mean" else ah.max(dim=0).values
        cls_vec = attn[0, 1:]
        v = (cls_vec - cls_vec.min()) / (cls_vec.max() - cls_vec.min() + 1e-8)
        vecs.append(v)

    if layer_mode == "last":
        acc = vecs[-1]
    elif layer_mode == "rollout":
        acc = vecs[0].clone()
        for v in vecs[1:]:
            acc = acc * v
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    elif layer_mode == "dynamic":
        acc = vecs[0].clone()
        for v in vecs[1:]:
            acc = acc + v * (1 - acc)
        acc = acc.clamp(0, 1)
    else:
        raise ValueError("layer_mode must be one of: last | dynamic | rollout")

    N = acc.numel()
    R = int(np.sqrt(N))
    assert R * R == N, f"Token grid not square: N={N}"
    hm = torch.nn.functional.interpolate(acc.view(R, R)[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    hm = hm.clamp(0, 1).cpu().numpy()

    cm = plt.get_cmap('jet')
    hm_u8 = (cm(hm) * 255).astype(np.uint8)
    fg = Image.fromarray(hm_u8).convert('RGBA')
    out_img = Image.blend(img_pil.convert('RGBA'), fg, alpha=0.5).convert("RGB")
    if flip_lr:
        out_img = out_img.transpose(Image.FLIP_LEFT_RIGHT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def compute_dataset_metrics_vitface_from_lists(
    feats_fn,
    model,
    size: int,
    ds_name: str,
    folders: Dict[str, List[pathlib.Path]],
    device: torch.device,
    amp: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute DU, DI, percentage ratio, and consistency for ViT-Face (CLS features)."""
    N = len(folders["upright_normal"])

    # probe layer count
    probe_tensor, _ = preprocess_vitface(folders["upright_normal"][0], size, device)
    N_LAYERS = len(feats_fn(probe_tensor))

    DU = np.zeros((N, N_LAYERS))
    DI = np.zeros((N, N_LAYERS))
    PR = np.zeros((N, N_LAYERS))

    for i in trange(N, desc=f"{ds_name}", leave=False):
        un, ut = preprocess_vitface(folders["upright_normal"][i],  size, device)[0], preprocess_vitface(folders["upright_thatcher"][i], size, device)[0]
        inn, it = preprocess_vitface(folders["inverted_normal"][i], size, device)[0], preprocess_vitface(folders["inverted_thatcher"][i], size, device)[0]
        with torch.autocast(device_type=device.type, enabled=amp):
            f_un, f_ut = feats_fn(un),  feats_fn(ut)
            f_in, f_it = feats_fn(inn), feats_fn(it)
        for b in range(N_LAYERS):
            du = np.linalg.norm((f_un[b] - f_ut[b]).detach().cpu().numpy())
            di = np.linalg.norm((f_in[b] - f_it[b]).detach().cpu().numpy())
            DU[i, b], DI[i, b] = du, di
            PR[i, b] = ((du / max(di, 1e-12)) - 1) * 100

    CD = (DU > DI).mean(axis=0) * 100
    return {"PR_mean": PR.mean(axis=0), "CD": CD, "N_LAY": N_LAYERS}


def run_vitface_pipeline(args, folders_by_ds: Dict[str, Dict[str, List[pathlib.Path]]]) -> None:
    device = torch.device(args.device)
    model = load_vitface(args.vitface_model, device)
    size = args.vitface_img_size

    outdir = pathlib.Path(args.out_vitface); outdir.mkdir(parents=True, exist_ok=True)
    saldir = outdir / "saliency"; saldir.mkdir(exist_ok=True)

    hook = _ViTFaceFeatHook(model)
    def feats_fn(x_bchw: torch.Tensor) -> List[torch.Tensor]:
        return hook(x_bchw)

    metrics = {}
    tag_map = {
        "Thatcher": {
            "upright_normal":   "normal_upright",
            "inverted_normal":  "normal_inverted",
            "upright_thatcher": "thatcher_upright",
            "inverted_thatcher":"thatcher_inverted",
        },
        "NonSemantic": {
            "upright_normal":   "NSLM_upright",
            "inverted_normal":  "NSLM_inverted",
            "upright_thatcher": "NSLM_upright",
            "inverted_thatcher":"NSLM_inverted",
        }
    }

    for name, folders in folders_by_ds.items():
        if not args.skip_cam:
            for cls, tag_base in tag_map[name].items():
                files = folders[cls][:max(0, args.cam_per_class)]
                for i, fp in enumerate(files):
                    tag = tag_base if args.cam_per_class == 1 else f"{tag_base}_{i+1}"
                    try:
                        flip = (name == "NonSemantic" and cls == "inverted_thatcher")
                        vitface_single_heatmap(model, fp, saldir / f"{tag}.png",
                                               size=size, head_reduce="mean", layer_mode="dynamic", flip_lr=flip)
                    except Exception as e:
                        print(f"[CAM:{name}/{cls}] {fp.name}: {e}")

        metrics[name] = compute_dataset_metrics_vitface_from_lists(
            feats_fn, model, size, name, folders, device, amp=args.amp
        )

    plot_and_csv(outdir, metrics, title="ViT-Face")
    print("✓ ViT-Face: Plots →", outdir)
    print("✓ ViT-Face: CSV   →", outdir / "metrics_layerwise.csv")
    print("✓ ViT-Face: Saliency overlays →", saldir)



# CLI
def main():
    p = argparse.ArgumentParser(description="Thatcher/NSLM analysis (new data layout) with VGG-16, VGG-Face, ViT, and ViT-Face")
    p.add_argument("--model", choices=["vgg16", "vggface", "vit", "vitface", "run-all"], required=True,
                   help="Which backbone(s) to run.")

    # Model paths / names
    p.add_argument("--vgg16-model",   type=pathlib.Path, default=DEF_VGG16_MODEL,
                   help="Path to custom VGG-16 checkpoint (.pth)")
    p.add_argument("--vggface-model", type=pathlib.Path, default=DEF_VGGFACE_MODEL,
                   help="Path to VGG-Face (Vgg_face_dag) checkpoint (.pth)")
    p.add_argument("--vit-arch", type=str, default="vit_base_patch8_224",
                   help="timm ViT architecture (e.g., vit_base_patch8_224)")
    p.add_argument("--vitface-model", type=pathlib.Path, default=DEF_VITFACE_MODEL,
                   help="Path to ViT-Face checkpoint (.pth)")

    # New unified data root (preferred)
    p.add_argument("--data-root", type=pathlib.Path, default=DEF_DATA_ROOT,
                   help="Unified dataset root (with upright_normal/, inverted_normal/, thatcher/{upright,inverted}, NSLM/{upright,inverted})")
    # Outputs
    p.add_argument("--out-vgg16",    type=pathlib.Path, default=DEF_OUT_VGG16,
                   help="Output dir for VGG-16 results")
    p.add_argument("--out-vggface",  type=pathlib.Path, default=DEF_OUT_VGGFACE,
                   help="Output dir for VGG-Face results")
    p.add_argument("--out-vit",      type=pathlib.Path, default=DEF_OUT_VIT,
                   help="Output dir for ViT results")
    p.add_argument("--out-vitface",  type=pathlib.Path, default=DEF_OUT_VITFACE,
                   help="Output dir for ViT-Face results")

    # Runtime
    p.add_argument("--img-size", type=int, default=224, help="Image size for VGG* models")
    p.add_argument("--vitface-img-size", type=int, default=112, help="Image size for ViT-Face")
    p.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast)")
    p.add_argument("--skip-cam", action="store_true", help="Skip saliency overlays")
    p.add_argument("--cam-per-class", type=int, default=1, help="How many images per class to overlay")
    args = p.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Build folder maps and lists
    layout = resolve_layout(data_root)
    folders_by_ds: Dict[str, Dict[str, List[pathlib.Path]]] = {}
    for ds_name, fmap in layout.items():
        # existence checks
        for k, d in fmap.items():
            if not d.exists():
                raise FileNotFoundError(f"[{ds_name}] missing directory: {d}")
        folders_by_ds[ds_name] = list_and_validate(fmap, ds_name)

    # Run pipelines
    if args.model in ("vgg16", "run-all"):
        run_vgg16_pipeline(args, folders_by_ds)
    if args.model in ("vggface", "run-all"):
        run_vggface_pipeline(args, folders_by_ds)
    if args.model in ("vit", "run-all"):
        run_vit_pipeline(args, folders_by_ds)
    if args.model in ("vitface", "run-all"):
        run_vitface_pipeline(args, folders_by_ds)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
