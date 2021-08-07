import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)

from tensorflow.python.ops.gen_dataset_ops import MapDataset, ParallelMapDataset
from model import generator
import os
import time
from tensorflow.keras.optimizers import Adam
from model.generator.generator import Generator
from model.discriminator.discriminator import Discriminator
from losses.losses import PixelLoss, ContentLoss
from losses.generator_loss import GeneratorLoss
from losses.discriminator_loss import DiscriminatorLoss
from lr_scheduler import MultiStepLR
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image_dataset_from_directory

W_PIXEL = 1e-2
W_FEATURE = 1.0
W_GAN = 5e-3
PIXEL_CITERION = 'l1'
FEATURE_CITERION = 'l1'
GAN_TYPE = 'ragan'
LR_G = 1e-4
LR_D = 1e-4
LR_STEPS = [50000, 100000, 200000, 300000]
LR_RATE = 0.5

ADAM_BETA_1_G = 0.9
ADAM_BETA_2_G = 0.99
ADAM_BETA_1_D = 0.9
ADAM_BETA_2_D = 0.99

LOGS = './logs/'
MODEL = './saved_model/'

DATA_HIGH_RES = './data/high_res/'
DATA_LOW_RES = './data/low_res/'

BATCH_SIZE = 2
LOW_RES_SIZE = (90, 120)
HI_RES_SIZE = (360, 480)

EPOCHS = 100

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataset() -> MapDataset:
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    hi_res_ds = image_dataset_from_directory(
        DATA_HIGH_RES, 
        label_mode=None, 
        image_size=HI_RES_SIZE, 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    low_res_ds = image_dataset_from_directory(
        DATA_LOW_RES,
        label_mode=None,
        image_size=LOW_RES_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((low_res_ds, hi_res_ds))
    dataset = dataset.map(lambda low_res, hi_res: (normalization(low_res), normalization(hi_res)))
    dataset = configure_for_performance(dataset)

    return dataset


if __name__ == '__main__':

    if not os.path.exists(LOGS):
        os.mkdir(LOGS)

    timestamp = str(time.time() * 1000)
    
    tensorboard_dir = os.path.join(LOGS, timestamp)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    if not os.path.exists(MODEL):
        os.mkdir(MODEL)

    save_path = os.path.join(MODEL, timestamp)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    # mirrored_strategy = tf.distribute.MirroredStrategy()

    dataset = get_dataset()
    # dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    generator = Generator(64)
    generator_model = generator.build()

    discriminator = Discriminator()
    discriminator_model = discriminator.build()

    generator_model.summary()
    discriminator_model.summary()

    gen_loss = GeneratorLoss(GAN_TYPE)
    discriminator_loss = DiscriminatorLoss(GAN_TYPE)

    pixel_loss = PixelLoss(PIXEL_CITERION)
    content_loss = ContentLoss(FEATURE_CITERION)

    lr_G = MultiStepLR(LR_G, LR_STEPS, LR_RATE)
    lr_D = MultiStepLR(LR_D, LR_STEPS, LR_RATE)

    optimizer_G = Adam(lr_G, ADAM_BETA_1_G, ADAM_BETA_2_G)
    optimizer_D = Adam(lr_D, ADAM_BETA_1_D, ADAM_BETA_2_D)


    @tf.function
    def train_step(x, y) -> None:
        with tf.GradientTape(persistent=True) as tape:
            super_res = generator_model(x, training=True)
            high_res_output = discriminator_model(y, training=True)
            super_res_output = discriminator_model(super_res, training=True)

            losses_G = {}
            losses_D = {}
            
            losses_G['regular'] = tf.reduce_sum(generator_model.losses)
            losses_D['regular'] = tf.reduce_mean(discriminator_model.losses)

            losses_G['pixel'] = W_PIXEL * pixel_loss(y, super_res)
            losses_G['feature'] = W_FEATURE * content_loss(y, super_res)

            losses_G['gan'] = W_GAN * gen_loss(high_res_output, super_res_output)
            losses_D['gan'] = discriminator_loss(high_res_output, super_res_output)

            total_loss_G = tf.math.add_n([loss for loss in losses_G.values()])
            total_loss_D = tf.math.add_n([loss for loss in losses_D.values()])

        grads_G = tape.gradient(total_loss_G, generator_model.trainable_variables)
        grads_D = tape.gradient(total_loss_D, discriminator_model.trainable_variables)

        optimizer_G.apply_gradients(zip(grads_G, generator_model.trainable_variables))
        optimizer_D.apply_gradients(zip(grads_D, discriminator_model.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    total_steps = len(os.listdir(os.path.join(DATA_LOW_RES, 'images'))) // BATCH_SIZE

    for epoch in range(EPOCHS):
        step = 0
        for batch_low, batch_high in dataset.take(total_steps):
            total_loss_G, total_loss_D, losses_G, losses_D = train_step(batch_low, batch_high)

            print(
                '{:03d}/{:03d}   {:02d}/{:02d}, Reg Loss G: {:6f}, Pix Loss G: {:6f}, Feature Loss G: {:6f}, GAN Loss G: {:6f}, Total Loss G: {:6f}, Reg Loss D: {:6f}, GAN Loss D: {:6f}, Total Loss D: {:6f}'
                .format(
                    epoch + 1, EPOCHS, step + 1, total_steps, 
                    losses_G['regular'], losses_G['pixel'], losses_G['feature'], losses_G['gan'], total_loss_G, 
                    losses_D['regular'], losses_D['gan'], total_loss_D
                    )
            )


            if step % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss_G/total_loss', total_loss_G, step=step)
                    tf.summary.scalar(
                        'loss_D/total_loss', total_loss_D, step=step)
                    for k, l in losses_G.items():
                        tf.summary.scalar('loss_G/{}'.format(k), l, step=step)
                    for k, l in losses_D.items():
                        tf.summary.scalar('loss_D/{}'.format(k), l, step=step)

                    tf.summary.scalar(
                        'learning_rate_G', optimizer_G.lr(step), step=step)
                    tf.summary.scalar(
                        'learning_rate_D', optimizer_D.lr(step), step=step)

            step += 1

    gen_save_path = os.path.join(save_path, 'generator')
    dis_save_path = os.path.join(save_path, 'discriminator')

    os.mkdir(gen_save_path)
    os.mkdir(dis_save_path)

    generator_model.save(gen_save_path)
    discriminator_model.save(dis_save_path)
