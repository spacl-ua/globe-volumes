## CNN based Detection of Globes for Volume Quantification in CT Orbits images

### Introduction
Fast and accurate quantification of globe volumes in the event of an ocular trauma can provide clinicians with valuable diagnostic information. A Convolutional Neural Network is presented here for predicting globe contours in CT images of the orbits and their subsequent volume quantification.

### Architecture
Figure shows the 2D Modified Residual UNET architecture (MRes-UNET2D) used in this work. The network uses high-resolution 2D axial CT images of the orbits as inputs and yields contours for the globes, which are then used to compute globe volumes. Additional training details are available in the manuscript referenced in the Citation section.

<img src="https://github.com/lunastra26/globe-volumes/blob/master/Images/Architecture.JPG" width="400">

### Predictions
Figure shows the globe contours predicted by MRes-UNET2D on axial CT images of the orbits. Interglobe volume difference (IGVD) between the left and the right globe volumes is also shown.  

<img src="https://github.com/lunastra26/globe-volumes/blob/master/Images/Predictions.JPG" width="400">

### Description
The following are available in the repository
1) A pretrained CNN model for predicting globe masks in axial CT images of the orbits
2) An evaluation script for testing new CT images
3) Utilities script


### Citation
If you use this CNN model in your work, please site the following:

Lavanya Umapathy, Blair Winegar, Lea MacKinnon, Michael Hill, Maria I Altbach, Joseph M Miller and Ali Bilgin, 
"*Fully Automated Segmentation of Globes for Volume Quantification in CT Orbits Images Using Deep Learning*", AJNR 2020.
