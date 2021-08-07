import os
import cv2
import numpy as np
import tensorflow as tf

QUANTIZED_MODEL = './quantized_models/model.tflite'
VIDEO = 'video.mp4'
INPUT_SIZE = (32, 32)
TEST = './test'

if __name__ == '__main__':
    if not os.path.exists(TEST):
        os.mkdir(TEST)

    original_dir = os.path.join(TEST, 'original')
    if not os.path.exists(original_dir):
        os.mkdir(original_dir)

    input_dir = os.path.join(TEST, 'input_image')
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    output_dir = os.path.join(TEST, 'output_image')
    if not os.path.join(output_dir):
        os.mkdir(output_dir)

    
    interpreter = tf.lite.Interpreter(QUANTIZED_MODEL)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(VIDEO)

    if (cap.isOpened() == False):
        exit()

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Original Video', frame)

            img = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
            input_img = tf.constant(img / 255., dtype=tf.float32)
            input_img = tf.expand_dims(input_img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_img)
            interpreter.invoke()

            tflite_result = interpreter.get_tensor(output_details[0]['index'])
            result = np.array(tflite_result[0] * 255, dtype=np.uint8)

            cv2.imwrite(os.path.join(original_dir, '{}.jpg'.format(i)), frame)
            cv2.imwrite(os.path.join(input_dir, '{}.jpg'.format(i)), img)
            cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(i)), result)

            i += 1
            
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    