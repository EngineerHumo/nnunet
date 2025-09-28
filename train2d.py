import argparse
import os
from typing import Optional

import torch.backends.cudnn as cudnn
import setproctitle
from dataset2d import Data
from Zig_RiR2d import ZRiR
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import visdom
except ImportError:  # pragma: no cover - optional dependency
    visdom = None


def _prepare_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3 and tensor.size(0) not in (1, 3):
        tensor = tensor[:3]
    if tensor.dim() == 3:
        tensor_min = tensor.min()
        tensor = tensor - tensor_min
        tensor_max = tensor.max()
        if tensor_max > 0:
            tensor = tensor / tensor_max
    return tensor.clamp(0.0, 1.0)


class VisdomReporter:
    def __init__(self, use_visdom: bool, server: str, port: int, env: str):
        self.enabled = use_visdom and visdom is not None
        self._viz: Optional['visdom.Visdom'] = None
        if self.enabled:
            self._viz = visdom.Visdom(server=server, port=port, env=env)
            if not self._viz.check_connection():
                print('Warning: Unable to connect to Visdom server. Visualisation disabled.')
                self.enabled = False
                self._viz = None

    def show_sample(self, prefix: str, image: torch.Tensor, predictions: torch.Tensor,
                    labels: Optional[torch.Tensor] = None) -> None:
        if not self.enabled or self._viz is None:
            return

        image_tensor = _prepare_image_tensor(image)
        self._viz.image(image_tensor, win=f'{prefix}_input', opts={'title': f'{prefix} Input'})

        if predictions.dim() == 4:
            predictions = predictions.squeeze(0)
        for cls_idx in range(predictions.size(0)):
            pred_tensor = _prepare_image_tensor(predictions[cls_idx])
            self._viz.image(pred_tensor, win=f'{prefix}_pred_class_{cls_idx}',
                            opts={'title': f'{prefix} Pred Class {cls_idx}'})

        if labels is not None:
            if labels.dim() == 4:
                labels = labels.squeeze(0)
            for cls_idx in range(min(labels.size(0), predictions.size(0))):
                label_tensor = _prepare_image_tensor(labels[cls_idx])
                self._viz.image(label_tensor, win=f'{prefix}_label_class_{cls_idx}',
                                opts={'title': f'{prefix} Label Class {cls_idx}'})
class CrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        if weights is not None:
            weights = torch.from_numpy(np.array(weights)).float().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if target.dim() == inputs.dim():
            target = target.float()
        else:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class loss(nn.Module):
    def __init__(self, model, args2):
        super(loss, self).__init__()
        self.model = model
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(args2.nclass)

    def forward(self, input, label, train, label_onehot=None):
        output = self.model(input)
        if train:
            dice_target = label_onehot if label_onehot is not None else label.long()
            loss = self.dice_loss(output, dice_target) + self.ce_loss(output, label.long())

            return loss, output

        else:
            return output


def build_segmentation_model(args2, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ZRiR(channels=[64, 128, 256, 512], num_classes=args2.nclass,
                 img_size=args2.crop_size[0], in_chans=args2.input_channels)
    return model.to(device)


def get_model(args2, device=None):
    segmentation_model = build_segmentation_model(args2, device=device)
    model = loss(segmentation_model, args2)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device if device is not None else torch.device('cpu')
    if model_device is None:
        model_device = device if device is not None else torch.device('cpu')
    return model.to(model_device)


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, warmup_iter=None, power=0.9):
    if warmup_iter is not None and cur_iters < warmup_iter:
        lr = base_lr * cur_iters / (warmup_iter + 1e-8)
    elif warmup_iter is not None:
        lr = base_lr * ((1-float(cur_iters - warmup_iter) / (max_iters - warmup_iter))**(power))
    else:
        lr = base_lr * ((1 - float(cur_iters / max_iters)) ** (power))
    optimizer.param_groups[0]['lr'] = lr


def log_validation_metrics(writer: Optional[SummaryWriter], summary: Optional[dict], epoch: int) -> None:
    if writer is None or summary is None:
        return

    average = summary.get('average')
    if average:
        for metric_name, value in average.items():
            writer.add_scalar(f'val/{metric_name}', value, epoch)

    per_class = summary.get('per_class', [])
    for entry in per_class:
        class_name = entry.get('Class', 'cls')
        for metric_name, value in entry.items():
            if metric_name == 'Class':
                continue
            writer.add_scalar(f'val/{class_name}/{metric_name}', value, epoch)


def export_trained_model_to_onnx(args2, checkpoint_path: str) -> None:
    if not args2.onnx_path:
        return

    if not os.path.exists(checkpoint_path):
        print(f'ONNX export skipped because checkpoint {checkpoint_path} was not found.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmentation_model = build_segmentation_model(args2, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    segmentation_model.load_state_dict(state_dict)
    segmentation_model.eval()

    if args2.export_input_size is not None:
        if len(args2.export_input_size) == 2:
            height, width = args2.export_input_size
        elif len(args2.export_input_size) == 3:
            _, height, width = args2.export_input_size
        else:
            raise ValueError('export_input_size must have 2 (H W) or 3 (C H W) values')
    else:
        height, width = args2.crop_size

    dummy_input = torch.randn(1, args2.input_channels, height, width, device=device)

    os.makedirs(os.path.dirname(args2.onnx_path) or '.', exist_ok=True)

    torch.onnx.export(
        segmentation_model,
        dummy_input,
        args2.onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=args2.onnx_opset,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )
    print(f'Exported ONNX model to {os.path.abspath(args2.onnx_path)}')


def train():
    from test2d import Eval
    args2 = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args2, device=device)

    data_train = Data(train=True, dataset=args2.dataset, crop_szie=args2.crop_size)
    dataloader_train = DataLoader(
        data_train,
        batch_size=args2.train_batchsize,
        shuffle=True,
        num_workers=args2.train_workers,
        pin_memory=True,
        drop_last=False,
        sampler=None)
    data_val = Data(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    dataloader_val = DataLoader(
        data_val,
        batch_size=args2.val_batchsize,
        shuffle=False,
        num_workers=args2.val_workers,
        pin_memory=True,
        sampler=None)

    use_visdom = not args2.disable_visdom
    vis_reporter = VisdomReporter(use_visdom, args2.visdom_server, args2.visdom_port, args2.visdom_env)

    if args2.val_output_dir:
        os.makedirs(args2.val_output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW([{'params':
                                        filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                    'lr': args2.lr}],
                                  lr=args2.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.0001,
                                  )


    writer = SummaryWriter(log_dir=args2.log_dir) if not args2.disable_logging else None

    checkpoint_dir = os.path.join('.', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_dice = -float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    last_checkpoint_path = os.path.join(checkpoint_dir, 'last_model.pth')

    global_step = 0
    for epoch in range(args2.end_epoch):
        model.train()
        setproctitle.setproctitle("Zig-RiR:" + str(epoch) + "/" + "{}".format(args2.end_epoch))

        for i, sample in enumerate(dataloader_train):
            image = sample['image'].cuda().float()
            label = sample['label'].cuda()
            label_indices = sample['label_indices'].cuda()

            if label_indices.dim() == 4 and label_indices.size(1) == 1:
                label_indices = label_indices.squeeze(1)
            label_indices = label_indices.long()

            label_onehot = None
            if label.dim() == 4 and label.size(1) == args2.nclass:
                label_onehot = label.float()


            losses, logits = model(image, label_indices, True, label_onehot)
            loss = losses.mean()

            lenth_iter = len(dataloader_train)
            adjust_learning_rate(optimizer,
                                args2.lr,
                                args2.end_epoch * lenth_iter,
                                i + epoch * lenth_iter,
                                args2.warm_epochs * lenth_iter
                                )
            print("epoch:[{}/{}], iter:[{}/{}], ".format(epoch, args2.end_epoch, i, len(dataloader_train)))
            if vis_reporter.enabled and i == 0:
                probs = torch.softmax(logits.detach(), dim=1)[0]
                if label_onehot is not None:
                    gt_vis = label_onehot[0].detach()
                else:
                    gt_vis = F.one_hot(label_indices[0].detach(), num_classes=args2.nclass).permute(2, 0, 1).float()
                vis_reporter.show_sample('train', image[0], probs, gt_vis)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

        if epoch % args2.val_interval == 0 or epoch == args2.end_epoch - 1:
            print('val num / batchsize:', len(dataloader_val))
            eval_results = Eval(dataloader_val, model, args2,
                                vis_reporter=vis_reporter,
                                epoch=epoch,
                                save_dir=args2.val_output_dir if args2.val_output_dir else None)
            if eval_results is not None:
                average_metrics = eval_results.get('average')
                if average_metrics is not None:
                    val_dice = average_metrics.get('Dice')
                    if val_dice is not None and val_dice > best_dice:
                        best_dice = val_dice
                        torch.save(model.model.state_dict(), best_checkpoint_path)
                        print(f'New best model saved with Dice={best_dice:.2f} at {best_checkpoint_path}')
                log_validation_metrics(writer, eval_results, epoch)
    torch.save(model.model.state_dict(), './weight.pkl')
    torch.save(model.model.state_dict(), last_checkpoint_path)

    if writer is not None:
        writer.close()

    export_trained_model_to_onnx(args2, best_checkpoint_path if os.path.exists(best_checkpoint_path) else last_checkpoint_path)



def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='prp')
    parser.add_argument("--end_epoch", type=int, default=400)

    parser.add_argument("--warm_epochs", type=int, default=5)

    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--train_batchsize", type=int, default=2)
    parser.add_argument("--val_batchsize", type=int, default=1)
    parser.add_argument("--train_workers", type=int, default=8, help='Number of worker processes for training loader')
    parser.add_argument("--val_workers", type=int, default=4, help='Number of worker processes for validation loader')
    parser.add_argument("--crop_size", type=int, nargs='+', default=[1024, 1024], help='H, W')

    parser.add_argument("--nclass", type=int, default=4)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--val_output_dir", type=str, default='./val_predictions',
                        help='Directory for saving validation predictions')
    parser.add_argument("--visdom_server", type=str, default='http://localhost',
                        help='Visdom server URL')
    parser.add_argument("--visdom_port", type=int, default=8097, help='Visdom server port')
    parser.add_argument("--visdom_env", type=str, default='zig_rir', help='Visdom environment name')
    parser.add_argument("--disable_visdom", action='store_true', help='Disable Visdom visualisation')
    parser.add_argument("--log_dir", type=str, default='./runs/zig_rir', help='TensorBoard log directory')
    parser.add_argument("--disable_logging", action='store_true', help='Disable TensorBoard logging')
    parser.add_argument("--val_interval", type=int, default=10, help='Number of epochs between validations')
    parser.add_argument("--onnx_path", type=str, default='./checkpoints/best_model.onnx', help='Output path for ONNX export')
    parser.add_argument("--onnx_opset", type=int, default=13, help='ONNX opset version')
    parser.add_argument("--export_input_size", type=int, nargs='+', default=None,
                        help='Optional ONNX export size. Provide H W or C H W')

    args2 = parser.parse_args()

    return args2



if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.enabled = True
    train()



