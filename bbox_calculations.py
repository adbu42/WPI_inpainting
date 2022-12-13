from __future__ import annotations
from typing import Tuple, NamedTuple
import math


class BBox(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_opencv_bounding_rect(cls, x, y, width, height):
        return cls(x, y, x + width, y + height)

    def top_left(self) -> Tuple:
        return self.left, self.top

    def bottom_right(self) -> Tuple:
        return self.right, self.bottom

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_points(self) -> Tuple:
        # top left, top right, bottom right, bottom left
        return self.top_left(), \
               (self.right, self.top), \
               self.bottom_right(), \
               (self.left, self.bottom)

    def is_overlapping_with(self, other_box: BBox):
        return self.left < other_box.right and \
               self.right > other_box.left and \
               self.top < other_box.bottom and \
               self.bottom > other_box.top

    def get_mutual_bbox(self, other_box):
        return BBox(
            min(self.left, other_box.left),
            min(self.top, other_box.top),
            max(self.right, other_box.right),
            max(self.bottom, other_box.bottom),
        )


def calculate_bboxes_for_patches(image_width: int, image_height: int, patch_overlap: int, patch_size: int) -> Tuple[BBox]:
    patches = []
    if patch_overlap is not None:
        current_x, current_y = (0, 0)
        while current_y < image_height:
            while current_x < image_width:
                image_box = BBox(current_x, current_y, current_x + patch_size,
                                 current_y + patch_size)
                patches.append(image_box)
                current_x += patch_size - patch_overlap
            current_x = 0
            current_y += patch_size - patch_overlap
    else:
        # automatic overlap calculation
        windows_in_width = math.ceil(image_width / patch_size)
        total_width_overlap = windows_in_width * patch_size - image_width
        windows_in_height = math.ceil(image_height / patch_size)
        total_height_overlap = windows_in_height * patch_size - image_height

        width_overlap_per_patch = total_width_overlap // windows_in_width
        height_overlap_per_patch = total_height_overlap // windows_in_height

        for y_idx in range(windows_in_height):
            start_y = int(y_idx * (patch_size - height_overlap_per_patch))
            for x_idx in range(windows_in_width):
                start_x = int(x_idx * (patch_size - width_overlap_per_patch))
                image_box = BBox(start_x, start_y, start_x + patch_size, start_y + patch_size)
                patches.append(image_box)

    return tuple(patches)
