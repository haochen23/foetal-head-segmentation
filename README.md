# Foetal Head Segmentation on Ultrasound Images - Gestational Age Prediction

## Notebooks

U-Net-224 with Residual blocks generates a large size model for accurate segmentation.

U-Net-96 generates a smaller sized segmentation model, for edge deployment on embedded systems.

## Usage:

First, clone this repo.

Second, move the foetal-head-us dataset to this repo.

Third, model training.

```
python3 train.py
```

Test the model on test set.
```
python3 predict_test.py
```
