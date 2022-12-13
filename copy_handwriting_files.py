from pathlib import Path
import json
from bbox_calculations import calculate_bboxes_for_patches
from PIL import Image
import argparse
import random


def main(args: argparse.Namespace):
    source_directory = args.source_directory
    target_directory = args.target_directory
    handwriting_bool_file = args.handwriting_file
    handwriting_list = []

    with open(handwriting_bool_file) as f:
        handwriting_analysis = json.load(f)
        for i in range(len(handwriting_analysis)):
            if not handwriting_analysis[i]['has_handwriting']:
                handwriting_list.append(Path(handwriting_analysis[i]['file_name']).name)

    for file_name in source_directory.rglob('*.jpg'):
        if Path(file_name).name in handwriting_list:
            if random.random() >= args.dataset_percentage:
                continue
            image_to_cut_into = Image.open(file_name)
            image_width, image_height = image_to_cut_into.size
            bboxes_for_patches = calculate_bboxes_for_patches(image_width, image_height, args.patch_overlap, args.patch_size)
            for patch_coordinates in bboxes_for_patches:
                patch = image_to_cut_into.crop(patch_coordinates)
                patch.save(Path(target_directory, f'{file_name.stem}-{patch_coordinates.left:05d}{patch_coordinates.right:05d}{patch_coordinates.top:05d}{patch_coordinates.bottom:05d}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy files from the wpi directory and cut them into patches.')
    parser.add_argument("-sd", "--source-directory", type=Path, default=None,
                        help="The Path to the source directory.")
    parser.add_argument("-td", "--target-directory", type=Path, default=None,
                        help="The Path to the target directory.")
    parser.add_argument("-hf", "--handwriting-file", type=Path, default=None,
                        help="The Path to the file where the handwriting classification is done.")
    parser.add_argument("-ps", "--patch-size", type=int, default=256,
                        help="The size of the patches to cut from the images.")
    parser.add_argument("-po", "--patch-overlap", type=int, default=None,
                        help="How much overlap the patches will have.")
    parser.add_argument("-dp", "--dataset-percentage", type=float, default=1.0,
                        help="What percentage of the source dataset is sampled and copied. Must be between 0 and 1")

    parsed_args = parser.parse_args()
    main(args=parsed_args)
