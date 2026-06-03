# Detection of brain tumors in MRI brain images

Alfonso Imanol Macias Marañon - A01808099
- Current model: [Model](https://drive.google.com/file/d/1V4-ZYqtuKhufF1wQ_Vg1VEozwEV6bqIH/view?usp=sharing) (~17.5 minutes)

## Project description

The main objective of this project is to use machine learning to accurately classify MRI images into two categories: healthy tissue and images containing tumors.

## Dataset description

The dataset used in this project was obtained from the [Kaggle](https://www.kaggle.com?utm_source=chatgpt.com) platform and consists of a collection of brain magnetic resonance imaging (MRI) scans designed for brain tumor detection. 

The dataset is divided into two main categories: images labeled **“yes”**, which contain abnormalities or spheroids associated with brain diseases and potential tumors, and images labeled **“no”**, which do not exhibit such abnormalities.

The complete dataset contains a total of 253 images, distributed as 155 images in the **“yes”** class and 98 images in the **“no”** class. 

## Data preprocessing

### Data transformation

The first step was to preprocess the MRI images in order to improve model performance and facilitate the training process. The following transformations were applied:
- Standardizing file names to facilitate visualization and iteration during data processing.
- Removal of the black borders surrounding the MRI scans.
- Normalization of all images to a fixed size of **256 × 256 pixels**.
- Conversion of image files to the **PNG** format to ensure consistent image quality.

These preprocessing steps were performed using the `cleandataset.py` script. The script processed the original Kaggle dataset by converting the images into pixel arrays, transforming them into grayscale representations, generating binary masks to identify the relevant brain region, and then cropping and resizing the images.

### Data splitting
```text
dataset__readytouse/
├── train/
│   ├── yes/
│   └── no/
├── validation/
│   ├── yes/
│   └── no/
└── test/
    ├── yes/
    └── no/
```
The dataset was manually split into three subsets: **training**, **testing**, and **validation**. The resulting directory structure was stored in the `dataset_readytouse` folder.

**Class: YES** (*N = 155*)

* Training: 93 images (60%)
* Testing: 31 images (20%)
* Validation: 31 images (20%)

**Class: NO** (*N = 98*)

* Training: 60 images (61.2%)
* Testing: 19 images (19.4%)
* Validation: 19 images (19.4%)

**Overall dataset:** 253 images.

The dataset was partitioned using an approximately balanced **60/20/20 split** across both classes, ensuring a similar class distribution in the training, testing, and validation subsets.

## Analysis Framework

| Category        | Description                            |
| --------------- | ------------------------------------ |
| Architecture    | VGG16 knowledge transfer CNN         |
| Evaluation      | Accuracy, Loss, F1-score               |
| Explainability  | Confussion Matrix               |

* Construction of convolutional block layers.
* Review of academic papers to produce solid decisition on backpropagation, optimization, and architecture.

## References

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385.

Springer Article (2025). DOI: 10.1186/s12929-025-01213-y.

Video Lecture:
https://www.youtube.com/watch?v=udaRL6NdItY

Video Lecture:
https://www.youtube.com/watch?v=12UvnLp-8qg
