import torch
import yaml
from pathlib import Path
from PIL import Image
import argparse
from analysis_segmenter import AnalysisSegmenter
from typing import Union
from tqdm import tqdm
from torchvision import transforms
from models.LBAMModel import LBAMModel
import torch.nn as nn
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='the directory of the images you want to remove the handwriting from')
parser.add_argument('--output-dir', type=Path, help='the directory of the images without handwriting')
args = parser.parse_args()


def is_image(file_name: Union[str, Path]) -> bool:
    if not isinstance(file_name, Path):
        file_name = Path(file_name)
    return file_name.suffix.lower() in ['.png', '.jpeg', '.jpg', '.svg']


def inpaint_patch(segmentation: Image, ground_truth: torch.Tensor, ) -> torch.Tensor:
    max_pooling = nn.MaxPool2d(7, 1, 3)
    to_tensor = transforms.ToTensor()
    segmentation_tensor = to_tensor(segmentation).cuda()
    mask = segmentation_tensor[0]
    ones = mask >= 0.5
    zeros = mask < 0.5
    mask.masked_fill_(ones, 1.0)
    mask.masked_fill_(zeros, 0.0)
    mask = mask.repeat(3, 1, 1)
    mask = max_pooling(mask)  # expand the masks so that the whole handwriting is covered
    mask = 1 - mask
    sizes = ground_truth.size()
    image = ground_truth * mask
    inputImage = torch.cat((image, mask[0].view(1, sizes[1], sizes[2])), 0)
    inputImage = inputImage.view(1, 4, sizes[1], sizes[2])
    mask = mask.view(1, sizes[0], sizes[1], sizes[2])

    netG = LBAMModel(4, 3)
    netG.load_state_dict(torch.load('weights/inpainting.pth'))
    for param in netG.parameters():
        param.requires_grad = False
    netG.eval()
    netG = netG.cuda()
    output = netG(inputImage, mask)
    output = output * (1 - mask) + inputImage[:, 0:3, :, :] * mask
    return output, mask


config_file = Path('application_config.yaml')
with config_file.open() as f:
    model_config = yaml.safe_load(f)

hyperparam_config = {'patch_overlap': model_config['patch_overlap'], 'min_confidence': model_config['min_confidence'],
                     'min_contour_area': model_config['min_contour_area']}

segmenter = AnalysisSegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=Path('/workspace/final_application/synthesis_in_style_lightning/stylegan_code_finder/handwriting_colors.json'),
        original_config_path=Path(model_config["config_path"]),
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=False,
        show_confidence_in_segmentation=False
    )

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
        inpainted_image, mask = inpaint_patch(color_prediction, patch['ground_truth'])
        mask_patches[patch_counter]['prediction'] = mask
        predicted_patches[patch_counter]['prediction'] = inpainted_image
    assembled_image = segmenter.assemble_predictions(predicted_patches, image.size)
    assembled_mask = segmenter.assemble_predictions(mask_patches, image.size)
    transforms.ToPILImage()(assembled_image).save(Path(args.output_dir, Path(image_path).name))
    transforms.ToPILImage()(assembled_mask).save(Path(args.output_dir, f'{Path(image_path).name}_mask.jpg'))
