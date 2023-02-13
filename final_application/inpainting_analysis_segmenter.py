from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Union

import torch
import torch.nn.functional as torch_functional
from PIL.Image import Image as ImageClass
from torchvision import transforms

from synthesis_in_style_lightning.stylegan_code_finder.segmentation.analysis_segmenter import AnalysisSegmenter


class InpaintingAnalysisSegmenter(AnalysisSegmenter):

    def __init__(self, model_checkpoint: str, device: str, class_to_color_map: Union[str, Path],
                 original_config_path: Optional[Path] = None, batch_size: Optional[int] = None,
                 max_image_size: int = None, print_progress: bool = True, patch_overlap: int = 0,
                 patch_overlap_factor: float = 0.0, show_confidence_in_segmentation: bool = False):
        super().__init__(model_checkpoint=model_checkpoint, device=device, class_to_color_map=class_to_color_map,
                         original_config_path=original_config_path, batch_size=batch_size,
                         max_image_size=max_image_size, print_progress=print_progress,
                         patch_overlap=patch_overlap, patch_overlap_factor=patch_overlap_factor,
                         show_confidence_in_segmentation=show_confidence_in_segmentation)
        self.patch_size = int(self.config['cutting_size'])

    def crop_and_batch_patches(self, input_image: ImageClass, normalize: bool = True) -> Iterator[dict]:
        if normalize:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            transform_list = [
                transforms.ToTensor()
            ]
        transform_list = transforms.Compose(transform_list)

        bboxes_for_patches = self.calculate_bboxes_for_patches(*input_image.size)
        for i in range(0, len(bboxes_for_patches), self.batch_size):
            batch_bboxes = bboxes_for_patches[i:i + self.batch_size] if self.batch_size > 1 else [bboxes_for_patches[i]]
            batch_images = [input_image.crop(bbox) for bbox in batch_bboxes]
            batch_images = [transform_list(image) for image in batch_images]
            batch_images = torch.stack(batch_images, dim=0)
            batch_images = batch_images.to(self.device)
            yield {'images': batch_images, 'bboxes': batch_bboxes}

    def predict_patches(self, patches: Iterator[dict]) -> [dict]:
        predicted_patches = []
        for batch in self.progress_bar(patches, desc="Predicting patches...", leave=False):
            batch_interpolated = torch_functional.interpolate(batch['images'],
                                                              (self.config['image_size'], self.config['image_size']))
            with torch.no_grad():
                prediction = self.network.predict(batch_interpolated)
            prediction = torch_functional.interpolate(prediction, (self.patch_size, self.patch_size))
            for i, bbox in enumerate(batch['bboxes']):
                predicted_patches.append({
                    "prediction": prediction[i],
                    "bbox": bbox,
                    "ground_truth": batch['images'][i]
                })

        return predicted_patches

    def assemble_predictions(self, patches: List[dict], output_size: Tuple) -> torch.Tensor:
        # dimensions are height, width, class for easier access
        num_classes = self.network.num_classes
        max_width = output_size[0]
        max_height = output_size[1]
        assembled_predictions = torch.full((max_height, max_width, num_classes), float("-inf"), device=self.device)

        for patch in self.progress_bar(patches, desc="Merging patches...", leave=False):
            reordered_patch = torch.squeeze(patch["prediction"]).permute(1, 2, 0)
            x_start, y_start, x_end, y_end = patch["bbox"]
            x_end = min(x_end, max_width)
            y_end = min(y_end, max_height)
            window_height = y_end - y_start
            window_width = x_end - x_start
            patch_without_padding = reordered_patch[:window_height, :window_width, :]
            assembled_predictions[y_start:y_end, x_start:x_end, :] = patch_without_padding

        return assembled_predictions.permute(2, 0, 1)  # permute so that the shape matches the original network output

    def segment_image(self, image: ImageClass) -> torch.Tensor:
        image = self.convert_image_to_correct_color_space(image)

        if self.max_image_size > 0 and any(side > self.max_image_size for side in image.size):
            image.thumbnail((self.max_image_size, self.max_image_size))

        patches = self.crop_and_batch_patches(image)
        predicted_patches = self.predict_patches(patches)

        return predicted_patches
