import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19

def PixelLoss(criterion='l1'):
    """pixel loss"""
    if criterion == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))


def ContentLoss(criterion='l1', output_layer=54, before_act=True):
    """content loss"""
    if criterion == 'l1':
        loss_func = tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        loss_func = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)

    if output_layer == 22:  # Low level feature
        pick_layer = 5
    elif output_layer == 54:  # Hight level feature
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))

    if before_act:
        vgg.layers[pick_layer].activation = None

    fea_extrator = tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)

    @tf.function
    def content_loss(hr, sr):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        preprocess_sr = preprocess_input(sr * 255.) / 12.75
        preprocess_hr = preprocess_input(hr * 255.) / 12.75
        sr_features = fea_extrator(preprocess_sr)
        hr_features = fea_extrator(preprocess_hr)

        return loss_func(hr_features, sr_features)

    return content_loss