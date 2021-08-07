import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

from PIL import Image as im
import cv2
import os
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from tensorflow.keras.preprocessing import image_dataset_from_directory

BATCH_SIZE = 16
LOW_RES_SIZE = 32
HI_RES_SIZE = 128

DATA_HIGH_RES = './data/high_res/'
DATA_LOW_RES = './data/low_res/'

X_4 = 'x4'

def rename_files():
    d = os.path.join(DATA_LOW_RES, 'images')
    for f in tqdm(os.listdir(d)):
        f_name = f.replace(X_4, '')
        os.rename(os.path.join(d, f), os.path.join(d, f_name))

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

if __name__ == '__main__':
    mirrored_strategy = tf.distribute.MirroredStrategy()
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    hi_res_ds = image_dataset_from_directory(
        DATA_HIGH_RES, 
        label_mode=None, 
        image_size=(HI_RES_SIZE, HI_RES_SIZE), 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    low_res_ds = image_dataset_from_directory(
        DATA_LOW_RES,
        label_mode=None,
        image_size=(LOW_RES_SIZE, LOW_RES_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((hi_res_ds, low_res_ds))
    dataset = dataset.map(lambda low_res, hi_res: (normalization(low_res), normalization(hi_res)))
    dataset = configure_for_performance(dataset)
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    for dist_inputs in dist_dataset:
        print(len(dist_inputs[0]))
        # print(type(dist_inputs[1]))

    # i = 0
    # for batch_high, batch_low in dataset.take(50):
    #     print(i)
    #     i += 1
    #     img_high = tfds.as_numpy(batch_high[0])
    #     img_low = tfds.as_numpy(batch_low[0])

    #     hr = np.array(img_high, dtype=np.uint8)
    #     lr = np.array(img_low, dtype=np.uint8)

    #     cv2.imshow('High', hr)
    #     cv2.imshow('Low', lr)

    #     key =  cv2.waitKey(0)
    #     if key == 27:
    #         break

    # cv2.destroyAllWindows()

