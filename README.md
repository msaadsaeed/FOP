# FOP (ICASSP 2022)
Official implementation of FOP method as described in "Fusion and Orthogonal Projection for Improved Face-Voice Association"
<p align="center">
  <img src="imgs/title.PNG" width="80%"/>
</p>

## Proposed Methodology
(Left) Overall method. Fundamentally, it is a two-stream pipeline which generates face and voice embeddings. We
propose fusion and orthogonal projection (FOP) mechanism (dotted red box). (Right) The architecture of multimodal fusion.
<p align="center"> 
  <img src="imgs/proposed_fop.jpg" width="60%"/>
  <img src="imgs/gmf.jpg" width="35%"/>
 </p>

  
## Installation
We have used python==3.6.5 and torch==1.8.0 for these experiments. It may not run on other versions of Python/Torch.
To install dependencies run:
```
pip install -r requirements.txt
```
For installation of Pytorch and CUDA (For GPU):
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```
## Feature Extraction

## Training and Testing

## Comparison
Cross-modal matching results: (Left) FOP vs other losses used in V-F methods. (Right) Our method vs SOTA methods.
<p align="center"> 
  <img src="imgs/nway_loss.jpg" width="40%"/>
  <img src="imgs/nway_sota.jpg" width="40%"/>
 </p>

## Citing FOP
```BibTeX
@article{fop_fusion,
  title={FUSION AND ORTHOGONAL PROJECTION FOR IMPROVED FACE-VOICE ASSOCIATION},
  author={Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue},
  journal={International Conference on Acoustics, Speech, and Signal Processing (ICASSP-22)},
  year={2022}
}
```
