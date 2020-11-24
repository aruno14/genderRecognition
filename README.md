# GenderRecognition
Very simple gender recognition using TensorFlow.

More details in Medium article: https://aruno14.medium.com/very-simple-gender-recognition-bd9e4ac33d19

## Dataset
* UTKFace Dataset: https://www.kaggle.com/jangedoo/utkface-new

## Folder structure
* /train_gender.py
* /train_gender_grayscale.py
* /UTKFace/*.jpg

## Model
### RGB
* Model type: MobileNetV2
* Input size: (48, 48, 3)
* Accuracy: 0.9588

### Grayscale
* Model type: MobileNetV2
* Input size: (48, 48, 1)
* Accuracy: 0.9265
