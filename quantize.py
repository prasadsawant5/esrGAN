import os
import tensorflow as tf

SAVED_MODEL_DIR = './saved_model/1626350864102.862/generator/'
LITE_MODEL_PATH = './quantized_models/'

if __name__ == '__main__':
    if not os.path.exists(LITE_MODEL_PATH):
        os.mkdir(LITE_MODEL_PATH)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    tflite_quant_model = converter.convert()

    # Save the model.
    with open(os.path.join(LITE_MODEL_PATH, 'model.tflite'), 'wb') as f:
        f.write(tflite_quant_model)