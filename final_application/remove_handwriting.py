import json
from pathlib import Path
from PIL.Image import Image
import argparse
from analysis_segmenter import VotingAssemblySegmenter
from typing import Union
from tqdm import tqdm
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, help='the directory of the images you want to remove the handwriting from')
parser.add_argument('--output-dir', type=Path, help='the directory of the images without handwriting')
args = parser.parse_args()


def is_image(file_name: Union[str, Path]) -> bool:
    if not isinstance(file_name, Path):
        file_name = Path(file_name)
    return file_name.suffix.lower() in Image.EXTENSION.keys()


config_file = Path('application_config.yaml')
with config_file.open() as f:
    model_config = json.load(f)

hyperparam_config = {'patch_overlap': model_config['patch_overlap'], 'min_confidence': model_config['min_confidence'],
                     'min_contour_area': model_config['min_contour_area']}

segmenter = VotingAssemblySegmenter(
        model_config["checkpoint"],
        device="cuda",
        original_config_path=Path(model_config["config_path"]),
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=False,
        show_confidence_in_segmentation=args.show_confidence
    )

segmentation_output = Path(args.output_dir, 'segmentation_output')
segmentation_output.mkdir(parents=True, exist_ok=True)

image_paths = [f for f in args.input_dir.rglob("*") if is_image(f)]
assert len(image_paths) > 0, "There are no images in the given directory."
segmenter.set_hyperparams(hyperparam_config)
all_patch_paths = []

for i, image_path in enumerate(tqdm(image_paths, desc="Segmenting images...", leave=False)):
    original_image = Image.open(image_path)
    image = original_image.convert("L")
    predicted_patches = segmenter.segment_image(image)
    for patch in predicted_patches:
        patch_path = Path(args.output_dir, 'segmentation_images',
                          f'{image_path.stem}{patch["bbox"].left:05d}{patch["bbox"].right:05d}{patch["bbox"].top:05d}{patch["bbox"].bottom:05d}.jpg')
        all_patch_paths.append({'path': patch_path, 'bbox': patch['bbox'],
                                'image_size': image.size, 'image_name': image_path.stem})
        save_image(patch['prediction'], patch_path)
