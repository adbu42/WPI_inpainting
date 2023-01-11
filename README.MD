# Annotation segmentation and removal
Removes the handwriting detected by [Synthesis in Style](https://github.com/hendraet/synthesis-in-style) by using the [LBAM inpainting model](https://github.com/Vious/LBAM_Pytorch).

## Files:  
### fill_handwriting_with_background.py:  
Removes the handwriting from documents by replacing the handwriting with the mean value of the document.  
Usage: fill_handwriting_with_background.py --input-dir "directory of the documents with handwriting" --output-dir "output directory"
### remove_handwriting.py
Removes the handwriting from documents by replacing the handwriting with the ouput of LBAM.  
Usage: remove_handwriting.py --input-dir "directory of the documents with handwriting" --output-dir "output directory"
### get_pnsr_and_ssim.py
Outputs the pnsr, ssim, mse and l1-loss between two direcories with documents.
Usage: get_pnsr_and_ssim.py --ground-truth "directory with the ground-truth images" --removed-handwriting "directory with the inpainted images"


## Acknowledgments
Codebase mainly taken from:
```
@InProceedings{Xie_2019_ICCV,
author = {Xie, Chaohao and Liu, Shaohui and Li, Chao and Cheng, Ming-Ming and Zuo, Wangmeng and Liu, Xiao and Wen, Shilei and Ding, Errui},
title = {Image Inpainting With Learnable Bidirectional Attention Maps},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}

@misc{bartz2022synthesis,
      title={Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data}, 
      author={Christian Bartz and Hendrik Rätz and Jona Otholt and Christoph Meinel and Haojin Yang},
      year={2022},
      eprint={2107.06777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
