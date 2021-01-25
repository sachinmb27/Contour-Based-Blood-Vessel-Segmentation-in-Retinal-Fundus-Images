# Contour Based Blood Vessel Segmentation in Retinal Fundus Images
This project involves segmenting blood vessels from retinal fundus images using OpenCV. This work was done as a part of an academic project.

## Abstract
Retinal image analysis plays an important part in identifying various eye related diseases such as Diabetic Retinopathy (DR), glaucoma and many others. Detecting these diseases in the early stage will reduce their severity and prevent eye blindness. Accurate segmentation of blood vessels plays an important part in identifying the retinal diseases at an early stage. Existing systems. In the proposed work, an unsupervised approach based on Contour detection has been proposed for effective segmentation of retinal blood vessels. The proposed Contour Based Blood Vessel Segmentation (CBBVS) method performs preprocessing using Contrast Limited Adaptive Histogram Equalization (CLAHE) followed by alternate sequential filtering to generate a noise-free image. The resultant image undergoes Otsu thresholding for candidate extraction followed by contour detection to properly segment the blood vessels. The CBBVS method has been tested on the DRIVE dataset and
the result shows that the proposed method achieved a sensitivity of 58.79%, specificity of 90.77% and accuracy of 86.7%.

## Data
The dataset is taken from the [STARE database](https://cecas.clemson.edu/~ahoover/stare/).

## Publication
A paper has been published based on this project in [International Journal of Applied Evolutionary Computation](https://www.igi-global.com/gateway/article/214896).
