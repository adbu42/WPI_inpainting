import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError, PeakSignalNoiseRatio
import torchvision.transforms.functional as F


def main(args: argparse.Namespace):
    ground_truth = args.ground_truth
    removed_handwriting = args.removed_handwriting

    ground_truth_images = {image_name.stem: F.to_tensor(Image.open(image_name).convert('RGB'))
                           for image_name in ground_truth.iterdir()}
    removed_handwriting_images = {image_name.stem: F.to_tensor(Image.open(image_name).convert('RGB'))
                                  for image_name in removed_handwriting.iterdir()}
    ground_truth_inpainted = [(ground_truth_images[image_name], removed_handwriting_images[image_name])
                              for image_name in ground_truth_images.keys()]

    ssim_function = StructuralSimilarityIndexMeasure()
    psnr_function = PeakSignalNoiseRatio()
    mse_function = MeanSquaredError()
    psnr = 0
    ssim = 0
    mse = 0
    l1_loss = 0
    image_count = len(ground_truth_inpainted)
    for gt, inpainted in ground_truth_inpainted:
        assert gt.size() == inpainted.size(), f'The images do not have the same size'
        mse += mse_function(gt, inpainted)
        psnr += psnr_function(gt, inpainted)
        ssim += ssim_function(torch.unsqueeze(gt, 0), torch.unsqueeze(inpainted, 0))
        l1_loss += nn.L1Loss()(gt, inpainted)
    psnr = psnr / image_count
    ssim = ssim / image_count
    mse = mse / image_count
    l1_loss = l1_loss / image_count

    print(f'mse: {mse}')
    print(f'psnr: {psnr}')
    print(f'ssim: {ssim}')
    print(f'l1_loss: {l1_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gets the MSE, L1-Loss, PSNR and SSIM by comparing two directories of images.')
    parser.add_argument('--ground-truth', type=Path, help='Path to the directory with ground_truth images')
    parser.add_argument('--removed-handwriting', type=Path,
                        help='Path to the directory with removed handwriting images')
    args = parser.parse_args()
    main(args=args)
