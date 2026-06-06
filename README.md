# Detection of brain tumors in MRI brain images

Alfonso Imanol Macias Marañon - A01808099

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

### Data augmentation

Due to the limited size of the dataset, data augmentation techniques were applied during training to increase data variability and improve the model's generalization ability. The following transformations were used:

* **Rotation:** Images were randomly rotated within a small range (±5% of a full rotation) using nearest-neighbor interpolation to preserve image continuity.
* **Translation:** Images were randomly shifted both horizontally and vertically by up to 5% of their dimensions.
* **Zoom:** Random zoom-in and zoom-out transformations of up to 10% were applied to simulate variations in image scale.
* **Contrast adjustment:** Image contrast was randomly modified by up to 10%, helping the model become more robust to intensity variations commonly found in MRI scans.

These augmentations were applied dynamically during training, generating slightly different versions of the images at each epoch and reducing the risk of overfitting.

## Model training # TO DO

All models were trained for 100 epochs with a batch size of 8. The loss function used was categorical crossentropy and the optimizer used was Adam. The learning rate was set to 0.00001. 

## Models # TO DO

### First Iteration # TO DO

[First Model Notebook](./classifier.ipynb)

#### Model description # TO DO

The first model was a sequential model with the following architecture:
- Conv2D layer with 32 filters, kernel size of (3, 3), and ReLU activation function
- MaxPooling2D layer with pool size of (2, 2)
- Conv2D layer with 32 filters, kernel size of (3, 3), and ReLU activation function
- MaxPooling2D layer with pool size of (2, 2)
- Flatten layer
- Dense layer with 256 units and ReLU activation function
- Dense layer with 90 units and softmax activation function

#### Results # TO DO

| Metric    | Train | Test |
|-----------|-------|------|
| Loss      | 26.5  | 22.5 |
| Accuracy  | 3.07  | 3.44 |
| Precision | -     | 0.26 |
| Recall    | -     | 0.22 |
| F1        | -     | 0.20 |
![Confusion matrix](./assets/conf_mat_4.png)

### Conclusions and next steps # TO DO

The model was able to learn some useful features, but the accuracy and F1 score are still very low. Most importantly, the test accuracy and loss are considerably worse than the training accuracy and loss. This indicates that the model is overfitting to the training data. The confusion matrix shows that the model is able to classify some classes correctly, but it is still very far from being able to classify all classes correctly and seems to predict porcupine more often, although it is still not significant enough.

Based on the observations the next proposed step is using transfer learning with the VGG16 model. The VGG16 model was trained on the ImageNet dataset, which has a similar number of classes and images. The idea is to use the weights of the VGG16 model as a starting point for the training of the new model. This should help the model to learn useful features from the images and improve the accuracy. This weights will be frozen to avoid overfitting, they will just be used for feature extraction.

The training accuracy and loss are worst than the previous model but the test metrics all improved, by a very low margin but improved nonetheless. The model did take more time to train (100 extra epochs) and there is still overfitting but it was considerably lower ad the training and testing metrics were closer. This could imply that further training will still be worth compared to the previous models that would just continue to overfit.

## Discussion # TO DO 

When compared with other attempts of image classification with this limited dataset, the results are quite underwhelming. As seen in [this notebook](https://www.kaggle.com/code/muhammadfaizan65/90-animals-image-classification-efficientnetb3/notebook) The author was able to achieve an F1 score of 0.94 and validation accuracy of 93%. This was achieved using the EfficientNetB3 model with transfer learning and data augmentation. It seem that the VGG16 model is not the best choice for this task. The EfficientNetB3 model is more recent and has been shown to outperform the VGG16 model in most tasks as shown in [2].

## References # TO DO , add hugging face's papers

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385.
[2]	Angelina, C. L., Xiao, F.-R., Vyas, S., Yang, P.-C., Chang, H.-T., & Luo, Y. (2026). Mod-SE(2): A geometric deep learning framework for brain tumor classification and segmentation in MRI images. Journal of Biomedical Science, 33, Article 11. https://doi.org/10.1186/s12929-025-01213-y
[3] Video Lecture: https://www.youtube.com/watch?v=udaRL6NdItY
[4] Video Lecture: https://www.youtube.com/watch?v=12UvnLp-8qg