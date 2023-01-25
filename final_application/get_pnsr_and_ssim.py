import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--ground-truth', type=Path, help='Path to the directory with ground_truth images')
parser.add_argument('--removed-handwriting', type=Path, help='Path to the directory with removed handwriting images')
args = parser.parse_args()

to_tensor = ToTensor()

ground_truth = args.ground_truth
removed_handwriting = args.removed_handwriting
ground_truth_images = {}
removed_handwriting_images = {}
for image_name in ground_truth.iterdir():
    ground_truth_images[image_name.stem] = to_tensor(Image.open(image_name).convert('RGB'))
for image_name in removed_handwriting.iterdir():
    removed_handwriting_images[image_name.stem] = to_tensor(Image.open(image_name).convert('RGB'))

ground_truth_inpainted = []
for image_name in ground_truth_images.keys():
    ground_truth_inpainted.append((ground_truth_images[image_name], removed_handwriting_images[image_name]))

ssim_function = StructuralSimilarityIndexMeasure()
psnr_function = PeakSignalNoiseRatio()
mse_function = MeanSquaredError()
psnr = 0
ssim = 0
mse = 0
l1_loss = 0
image_count = len(ground_truth_inpainted)
for gt_and_inpainted in ground_truth_inpainted:
    mse += mse_function(gt_and_inpainted[0], gt_and_inpainted[1])
    psnr += psnr_function(gt_and_inpainted[0], gt_and_inpainted[1])
    ssim += ssim_function(torch.unsqueeze(gt_and_inpainted[0], 0), torch.unsqueeze(gt_and_inpainted[1], 0))
    l1_loss += nn.L1Loss()(gt_and_inpainted[0], gt_and_inpainted[1])
psnr = psnr / image_count
ssim = ssim / image_count
mse = mse / image_count
l1_loss = l1_loss / image_count

print(f'mse: {mse}')
print(f'psnr: {psnr}')
print(f'ssim: {ssim}')
print(f'l1_loss: {l1_loss}')
