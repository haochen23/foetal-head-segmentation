import config
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from metrics import dice_coef, dice_coef_loss
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt 

smooth = config.smooth
path_test = config.PATH_TEST
IMG_HEIGHT = config.IMG_HEIGHT
IMG_WIDTH = config.IMG_WIDTH
test_list = os.listdir(path_test)

model = load_model(config.best_model_path, 
                   custom_objects={"dice_coef_loss":dice_coef_loss, "dice_coef":dice_coef})

X_test = np.empty((len(test_list), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
for i, item in enumerate(test_list):
    image = cv2.imread("foetal-head-us/test/" + item, 0)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
    X_test[i] = image
X_test = X_test[:,:,:,np.newaxis] / 255

y_pred = model.predict(X_test)

image_count = 0

for test, pred in zip(X_test[180:190],y_pred[180:190]):
    fig, ax = plt.subplots(1,3,figsize = (16,12))
    test = test.reshape((IMG_HEIGHT,IMG_WIDTH))

    pred = pred.reshape((IMG_HEIGHT,IMG_WIDTH))
    pred = pred>0.5
    pred = np.ma.masked_where(pred == 0, pred)
    ax[0].imshow(test, cmap = 'gray')

    ax[1].imshow(pred, cmap = 'gray')

    ax[2].imshow(test, cmap = 'gray', interpolation = 'none')
    ax[2].imshow(pred, cmap = 'jet', interpolation = 'none', alpha = 0.7)
    
    plt.savefig('results/result{}.png'.format(image_count))
    image_count += 1