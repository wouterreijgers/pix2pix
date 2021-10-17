# import os
# import tensorflow as tf
# from src.generator import *
# from src.discriminator import *
#
# generator = Generator_()
# discriminator = Discriminator_()
#
# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#
#
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#
# """## Testing on the entire test dataset"""
#
# # Run the trained model on the entire test dataset
# generate_images(generator, inp, tar, base_path='img/0', epoch=epoch)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
