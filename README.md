# FOP (ICASSP 2022)
Official implementation of FOP method as described in "Fusion and Orthogonal Projection for Improved Face-Voice Association"
<p align="center">
  <img src="title.PNG" width="30%"/>
</p>

## Proposed Methodology
(Left) Overall method. Fundamentally, it is a two-stream pipeline which generates face and voice embeddings. We
propose fusion and orthogonal projection (FOP) mechanism (dotted red box). (Right) The architecture of multimodal fusion.
<p align="center"> 
  <img src="imgs/proposed_fop.jpg" width="60%"/>
  <img src="imgs/gmf.jpg" width="35%"/>
 </p>

  
## Requirements

## Comparison
Cross-modal matching results: (Left) FOP vs other losses used in F-V methods. (Right) Our method vs SOTA methods.
<p align="center"> 
  <img src="imgs/nway_loss.jpg" width="30%"/>
  <img src="imgs/nway_sota.jpg" width="30%"/>
 </p>

## Citing FOP
@article{sasnet,
  title={FUSION AND ORTHOGONAL PROJECTION FOR IMPROVED FACE-VOICE ASSOCIATION},
  author={Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue},
  journal={Internation Conference on Acoustics, Speech, and Signal Processing (ICASSP-22)},
  year={2022}
}
