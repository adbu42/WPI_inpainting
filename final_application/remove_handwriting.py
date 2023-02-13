import argparse
import copy
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from inpainting_analysis_segmenter import InpaintingAnalysisSegmenter
from models.LBAMModel import LBAMModel


def is_image(file_name: Union[str, Path]) -> bool:
    if not isinstance(file_name, Path):
        file_name = Path(file_name)
    return file_name.suffix.lower() in ['.png', '.jpeg', '.jpg', '.svg']


def inpaint_patch(segmentation: Image, ground_truth: torch.Tensor, image_mean: bool, inpainting_model: nn.Module) -> torch.Tensor:
    # A kernel size of 7 is big enough to cover most handwriting.
    # A padding of 3 makes sure that the image size is not different.
    max_pooling = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

    # Map the values between 0 and 1 to 0 and 1
    segmentation_tensor = F.to_tensor(segmentation).cuda()
    mask = torch.zeros_like(segmentation_tensor[0])
    mask.masked_fill_(segmentation_tensor[0] >= 0.5, 1.0)

    # repeat the mask, so it covers all three channel of the input image
    # enlarge the masks so that the whole handwriting is covered
    mask = 1 - max_pooling(mask.repeat(3, 1, 1))
    image = ground_truth * mask
    # the LBAM-model needs a mask as a channel of the input image
    input_image = torch.cat((image, mask[0].unsqueeze(0)), 0)

    # unsqueeze image and mask to simulate a batch of one
    input_image = input_image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    if image_mean:
        # inpaint with image mean
        image_mean = torch.mean(ground_truth, dim=[1, 2])
        output = image_mean.unsqueeze(1).unsqueeze(2).repeat(1, 256, 256)
    else:
        # inpaint with the LBAM-model
        for param in inpainting_model.parameters():
            param.requires_grad = False
        inpainting_model.eval()
        inpainting_model = inpainting_model.cuda()
        output = inpainting_model(input_image, mask)

    # fill the masked parts of the image with the inpainting model, while the unmasked parts stay the same
    output = output * (1 - mask) + input_image[:, 0:3, :, :] * mask
    return output, mask


def main(args: argparse.Namespace):
    config_file = Path('application_config.yaml')
    with config_file.open() as f:
        model_config = yaml.safe_load(f)

    hyperparam_config = {'patch_overlap': model_config['patch_overlap'],
                         'min_confidence': model_config['min_confidence'],
                         'min_contour_area': model_config['min_contour_area']}

    segmenter = InpaintingAnalysisSegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=Path(
            '/workspace/final_application/synthesis_in_style_lightning/stylegan_code_finder/handwriting_colors.json'),
        original_config_path=Path(model_config["config_path"]),
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=False,
        show_confidence_in_segmentation=False
    )
    net_g = LBAMModel(4, 3)
    net_g.load_state_dict(torch.load('weights/inpainting.pth'))

    image_paths = [f for f in args.input_dir.rglob("*") if is_image(f)]
    assert len(image_paths) > 0, "There are no images in the given directory."
    segmenter.set_hyperparams(hyperparam_config)

    for i, image_path in enumerate(tqdm(image_paths, desc="Segmenting and inpainting images...", leave=False)):
        original_image = Image.open(image_path)
        image = original_image.convert("L")
        predicted_patches = segmenter.segment_image(image)
        mask_patches = copy.deepcopy(predicted_patches)
        original_batches = segmenter.crop_and_batch_patches(original_image, False)
        original_patches = []
        for original_batch in original_batches:
            for original_patch in original_batch['images']:
                original_patches.append(original_patch)
        for patch_counter, (patch, original_patch) in enumerate(zip(predicted_patches, original_patches)):
            color_prediction = segmenter.prediction_to_color_image(patch['prediction'])
            patch['ground_truth'] = original_patch
            inpainted_image, mask = inpaint_patch(color_prediction, patch['ground_truth'], args.image_mean_method, net_g)
            mask_patches[patch_counter]['prediction'] = mask
            predicted_patches[patch_counter]['prediction'] = inpainted_image
        assembled_image = segmenter.assemble_predictions(predicted_patches, image.size)
        assembled_mask = segmenter.assemble_predictions(mask_patches, image.size)
        F.to_pil_image(assembled_image).save(args.output_dir/image_path.name)
        F.to_pil_image(assembled_mask).save(args.output_dir/f'{image_path.name}_mask.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Removes handwriting from documents.')
    parser.add_argument('--input-dir', type=Path,
                        help='the directory of the images you want to remove the handwriting from')
    parser.add_argument('--output-dir', type=Path, help='the directory of the images without handwriting')
    parser.add_argument('--image-mean-method', type=bool, default=False,
                        help='If the image mean method should be used instead of LBAM')
    args = parser.parse_args()
    main(args=args)
