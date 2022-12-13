import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import NoReturn
from PIL import Image
from PIL.Image import Image
from tqdm import tqdm

from pytorch_training.images import is_image
from segmentation.analysis_segmenter import VotingAssemblySegmenter
from utils.image_utils import resize_image


def create_hyperparam_configs(args):
    overlap = list(itertools.product(args.absolute_patch_overlap, args.patch_overlap_factor))
    hyperparam_combinations = list(itertools.product(args.min_confidence, args.min_contour_area, overlap))

    hyperparam_names = [("min_confidence", "min_contour_area", "patch_overlap")] * len(hyperparam_combinations)
    hyperparam_configs = tuple(map(lambda x, y: {k: v for k, v in zip(x, y)},
                                   hyperparam_names, hyperparam_combinations))
    return hyperparam_configs


def preprocess_images(image: Image, args: argparse.Namespace) -> Image:
    if args.resize:
        image = resize_image(image, args.resize)
    if args.convert_to_black_white:
        image = image.convert("L")
    return image


def main(args: argparse.Namespace) -> NoReturn:
    root_dir = Path(__file__).resolve().parent.parent
    with args.config_file.open() as f:
        model_config = json.load(f)
    segmenter = VotingAssemblySegmenter(
        model_config["checkpoint"],
        device="cuda",
        class_to_color_map=root_dir / model_config["class_to_color_map"],
        original_config_path=Path(model_config["config_path"]),
        max_image_size=int(model_config.get("max_image_size", 0)),
        print_progress=False,
        show_confidence_in_segmentation=args.show_confidence
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [f for f in args.image_dir.glob("**/*") if is_image(f)]
    assert len(image_paths) > 0, "There are no images in the given directory."
    for hyperparam_config in tqdm(hyperparam_configs, desc="Processing hyperparameter configs", leave=True):
        segmenter.set_hyperparams(hyperparam_config)

        if evaluate:
            results["runs"].append(defaultdict(dict))

        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images...", leave=False)):
            original_image = Image.open(image_path)
            image = preprocess_images(original_image, args)
            assembled_prediction = segmenter.segment_image(image)
