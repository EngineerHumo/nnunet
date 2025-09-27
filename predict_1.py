import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset2d import Data
from train2d import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained 2D segmentation model using weight.pkl')
    parser.add_argument('--dataset', type=str, default='prp', help='Dataset name defined in dataset2d.Data')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Root directory that contains the dataset folders')
    parser.add_argument('--weight-path', type=str, default='./weight.pkl', help='Path to the trained model weights saved as a pickle')
    parser.add_argument('--output-dir', type=str, default='/output', help='Directory where predicted masks will be written')
    parser.add_argument('--onnx-path', type=str, default='./test1.onnx', help='File path to export the ONNX model snapshot')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[512, 512], help='Input size H W expected by the network')
    parser.add_argument('--nclass', type=int, default=4, help='Number of segmentation classes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version for export')
    return parser.parse_args()


def _normalize_class_to_gray(mask: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    scale = 255 // (num_classes - 1)
    return (mask.astype(np.uint8) * scale).clip(0, 255).astype(np.uint8)


def _names_from_batch(names, batch_index: int, batch_size: int) -> List[str]:
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names):
        return list(names)
    return [f'sample_{batch_index:04d}_{idx}' for idx in range(batch_size)]


def save_predictions(segmentation_model: torch.nn.Module,
                     dataloader: DataLoader,
                     device: torch.device,
                     output_dir: str,
                     num_classes: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    segmentation_model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device).float()
            names = batch.get('name')

            logits = segmentation_model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            resolved_names = _names_from_batch(names, batch_idx, predictions.shape[0])

            for idx, name in enumerate(resolved_names):
                pred_map = predictions[idx]
                gray_map = _normalize_class_to_gray(pred_map, num_classes)
                save_path = os.path.join(output_dir, f'{name}.png')
                Image.fromarray(gray_map, mode='L').save(save_path)


def export_onnx(segmentation_model: torch.nn.Module,
                onnx_path: str,
                input_channels: int,
                crop_size: Optional[List[int]],
                opset: int) -> None:
    if not onnx_path:
        return

    directory = os.path.dirname(onnx_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if crop_size is None or len(crop_size) < 2:
        raise ValueError('crop_size must provide width and height for ONNX export')

    height, width = crop_size[1], crop_size[0]

    model_device = next(segmentation_model.parameters()).device
    if model_device.type != 'cuda':
        if torch.cuda.is_available():
            segmentation_model = segmentation_model.to('cuda')
            model_device = torch.device('cuda')
        else:
            print('CUDA is not available; skipping ONNX export for CUDA-only model.')
            return

    segmentation_model = segmentation_model.eval()
    dummy_input = torch.randn(1, input_channels, height, width, device=model_device)

    torch.onnx.export(
        segmentation_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_test = Data(base_dir=args.data_dir, train=False, dataset=args.dataset, crop_szie=args.crop_size)
    input_channels = 3
    if len(data_test) > 0:
        sample = data_test[0]
        input_channels = sample['image'].shape[0]

    dataloader_test = DataLoader(
        data_test,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    model = get_model(args, device=device)
    state_dict = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    segmentation_model = model.model.to(device).eval()

    save_predictions(segmentation_model, dataloader_test, device, args.output_dir, args.nclass)
    print(f'Predictions saved to {os.path.abspath(args.output_dir)}')

    export_onnx(segmentation_model, args.onnx_path, input_channels, args.crop_size, args.opset)
    print(f'ONNX model exported to {os.path.abspath(args.onnx_path)}')


if __name__ == '__main__':
    main()
