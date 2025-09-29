"""Utility for exporting trained nnU-Net v2 models to ONNX.

This script inspects a trained experiment folder (the directory that contains
``dataset.json``/``plans.json`` and ``fold_*`` sub-directories) and exports the
stored PyTorch checkpoints to ONNX for standalone inference. The code mirrors
nnU-Net's own loading logic so that architecture, preprocessing plans and fold
selection stay consistent with the original training setup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _resolve_device(requested: str) -> torch.device:
    """Resolve the torch.device requested by the user."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA was requested but no GPU is available. Falling back to CPU for export.")
        return torch.device("cpu")
    return device


def _auto_detect_folds(model_dir: str, checkpoint_name: str) -> List[int]:
    """Use nnUNet's auto-detection logic to find available folds."""
    folds = nnUNetPredictor.auto_detect_available_folds(model_dir, checkpoint_name)
    folds = sorted({int(f) for f in folds})
    if not folds:
        raise RuntimeError(
            f"No folds with '{checkpoint_name}' found under {model_dir}. Please check the path.")
    return folds


def _collect_requested_folds(
    model_dir: str, folds: Optional[Sequence[str]], checkpoint_name: str
) -> List[int]:
    if folds is None or len(folds) == 0:
        return _auto_detect_folds(model_dir, checkpoint_name)

    collected: List[int] = []
    for fold in folds:
        if fold.lower() == "all":
            collected.extend(_auto_detect_folds(model_dir, checkpoint_name))
            continue
        try:
            collected.append(int(fold))
        except ValueError as exc:
            raise ValueError(f"Unable to interpret fold identifier '{fold}'.") from exc

    # deduplicate while preserving order of first occurrence
    ordered_unique: List[int] = []
    for fold in collected:
        if fold not in ordered_unique:
            ordered_unique.append(fold)
    return ordered_unique


def _load_network_from_checkpoint(
    model_dir: str,
    fold: int,
    checkpoint_name: str,
):
    checkpoint_path = join(model_dir, f"fold_{fold}", checkpoint_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found for fold {fold} under {model_dir}.")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)

    plans = load_json(join(model_dir, "plans.json"))
    dataset_json = load_json(join(model_dir, "dataset.json"))

    plans_manager = PlansManager(plans)
    configuration_name = checkpoint["init_args"]["configuration"]
    configuration_manager = plans_manager.get_configuration(configuration_name)

    trainer_name = checkpoint["trainer_name"]
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(
            f"Unable to locate trainer class '{trainer_name}'. Ensure the trainer is available on PYTHONPATH.")

    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"])
    network.eval()

    patch_size = configuration_manager.patch_size
    return network, patch_size, num_input_channels


def _make_dummy_input(
    num_input_channels: int, patch_size: Sequence[int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a dummy tensor that matches the expected network input shape."""
    spatial_dims = tuple(int(d) for d in patch_size)
    if len(spatial_dims) not in (2, 3):
        raise RuntimeError(
            "Unexpected dimensionality for patch size. Expected 2D or 3D data.")

    shape = (1, num_input_channels, *spatial_dims)
    return torch.randn(shape, device=device, dtype=dtype)


def export_fold_to_onnx(
    model_dir: str,
    fold: int,
    checkpoint_name: str,
    output_file: Path,
    device: torch.device,
    opset: int,
    dynamic_axes: bool,
    use_half_precision: bool,
) -> None:
    network, patch_size, num_input_channels = _load_network_from_checkpoint(model_dir, fold, checkpoint_name)

    target_dtype = torch.float16 if use_half_precision and device.type == "cuda" else torch.float32
    network = network.to(device=device, dtype=target_dtype)

    dummy_input = _make_dummy_input(num_input_channels, patch_size, device, target_dtype)

    dynamic_axes_map = None
    if dynamic_axes:
        # Allow dynamic batch and spatial dimensions
        axis_template = {0: "batch", 1: "channels"}
        for idx in range(len(patch_size)):
            axis_template[idx + 2] = f"dim{idx}"
        dynamic_axes_map = {"input": axis_template, "output": axis_template.copy()}

    output_file = output_file.with_suffix(".onnx")
    maybe_mkdir_p(str(output_file.parent))

    with torch.no_grad():
        torch.onnx.export(
            network,
            dummy_input,
            str(output_file),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_map,
            do_constant_folding=True,
        )

    print(f"Exported fold {fold} to {output_file}")

    empty_cache(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained nnU-Net v2 models to ONNX.")
    parser.add_argument(
        "model_dir",
        help="Path to the trained model directory (contains plans.json, dataset.json, fold_* folders).",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        default=None,
        help="Specific fold IDs to export (e.g. 0 1 2 3 4). Use 'all' or omit to export every available fold.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_final.pth",
        help="Checkpoint file name to export (default: checkpoint_final.pth).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for ONNX files. Defaults to <model_dir>/onnx_exports.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use for export.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run the export on (e.g. 'cpu', 'cuda:0', 'auto').",
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Export the ONNX graph with dynamic batch and spatial dimensions.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export using float16 precision (only if CUDA is available).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folds for which an ONNX file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Model directory '{model_dir}' does not exist.")

    folds = _collect_requested_folds(model_dir, args.folds, args.checkpoint)
    device = _resolve_device(args.device)

    output_dir = Path(args.output_dir) if args.output_dir else Path(model_dir) / "onnx_exports"
    maybe_mkdir_p(str(output_dir))

    for fold in folds:
        output_file = output_dir / f"fold_{fold}_{Path(args.checkpoint).stem}"
        if args.skip_existing and output_file.with_suffix(".onnx").is_file():
            print(f"Skipping fold {fold} (ONNX file already exists).")
            continue

        export_fold_to_onnx(
            model_dir=model_dir,
            fold=fold,
            checkpoint_name=args.checkpoint,
            output_file=output_file,
            device=device,
            opset=args.opset,
            dynamic_axes=args.dynamic_axes,
            use_half_precision=args.half,
        )

    print("Export complete.")


if __name__ == "__main__":
    main()