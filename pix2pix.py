import os
import time

from src import images as img
from src.data import load_image_test, load_image_train
from src import p2pGenerator
from src import discriminator
from src import losses
import tensorflow as tf
import matplotlib.pyplot as plt
from src.losses import generator_loss, discriminator_loss




PATH = "C:/Users/Wouter Reijgersberg/Documents/MAP/cityscapes/cityscapes/"

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

inp, re = img.load(PATH + 'train/100.jpg')

train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH + 'val/*.jpg')
# shuffling so that for every epoch a different image is generated
# to predict and display the progress of our model.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

"""## Build the Generator
  * The architecture of generator is a modified U-Net.
  * Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
  * Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
  * There are skip connections between the encoder and decoder (as in U-Net).
"""

# OUTPUT_CHANNELS = 3
down_model = p2pGenerator.downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)

up_model = p2pGenerator.upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

generator = p2pGenerator.Generator_()

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
plt.close()

"""## Build the Discriminator
  * The Discriminator is a PatchGAN.
  * Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
  * The shape of the output after the last layer is (batch_size, 30, 30, 1)
  * Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
  * Discriminator receives 2 inputs.
    * Input image and the target image, which it should classify as real.
    * Input image and the generated image (output of generator), which it should classify as fake.
    * We concatenate these 2 inputs together in the code (`tf.concat([inp, tar], axis=-1)`)
"""
"""
Build a class version of the discriminator here."""

discriminator = discriminator.Discriminator_()
disc_out = discriminator(inp[tf.newaxis, ...], gen_output, training=False)

"""To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

## Define the loss functions and the optimizer

* **Discriminator loss**
  * The discriminator loss function takes 2 inputs; **real images, generated images**
  * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
  * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
  * Then the total_loss is the sum of real_loss and the generated_loss

* **Generator loss**
  * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
  * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
  * This allows the generated image to become structurally similar to the target image.
  * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).
"""

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# accuracy_object = tf.keras.metrics.categorical_accuracy()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

"""## Checkpoints (Object-based saving)"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 200


def generate_images(model, test_input, tar, base_path='', epoch=0):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    """ Check the base path and make dir if needed."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 8))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(PATH + 'result/result_{}.png'.format(epoch))
    plt.close('all')


@tf.function
def train_step(input_image, target, g_loss_metric, d_loss_metric):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(input_image, target, training=True)
        disc_generated_output = discriminator(input_image, gen_output, training=True)

        gen_loss = generator_loss(loss_object, disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(loss_object,disc_real_output, disc_generated_output)

    # Update the metrics
    g_loss_metric.update_state(gen_loss)
    d_loss_metric.update_state(disc_loss)
    # accuracy_metric.update_state(tf.ones_like(gen_output), gen_output)
    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


print("starting training")


def train(dataset, epochs):
    # Create the metrics
    g_loss_log = []
    d_loss_log = []
    g_loss_metric = tf.keras.metrics.Mean(name='g_train_loss')
    d_loss_metric = tf.keras.metrics.Mean(name='d_train_loss')
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    for epoch in range(epochs):
        start = time.time()
        # Reset the metrics
        g_loss_metric.reset_states()
        d_loss_metric.reset_states()
        counter = 0
        for input_image, target in dataset:
            train_step(input_image, target, g_loss_metric, d_loss_metric)
            counter += 1
            if counter > 100:
                break
        # Get the metric results
        g_mean_loss = g_loss_metric.result()
        d_mean_loss = d_loss_metric.result()
        g_loss_log.append([g_mean_loss])
        d_loss_log.append([d_mean_loss])
        # mean_accuracy = accuracy_metric.result()
        #
        print('Epoch: ', epoch)
        print('  loss (g) (d) (g+d):     {:.3f}, {:.3f}, {:.3f}'.format(g_mean_loss, d_mean_loss,
                                                                        g_mean_loss + d_mean_loss))

        if (epoch) % 10 == 0:
            for i, (inp, tar) in enumerate(test_dataset.take(5)):
                generate_images(generator, inp, tar, base_path='results/{}'.format(i), epoch=epoch)
                losses.plot_losses(g_loss_log, d_loss_log, epoch=epoch, dataset='dataset')

        # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #   checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))


train(train_dataset, EPOCHS)

"""## Restore the latest checkpoint and test"""

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Testing on the entire test dataset"""

my_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
# shuffling so that for every epoch a different image is generated
# to predict and display the progress of our model.
# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
my_dataset = my_dataset.map(load_image_test)
my_dataset = my_dataset.batch(1)


for epoch, (inp, tar) in enumerate(my_dataset):
    generate_images(generator, inp, tar, base_path='img/0', epoch= epoch + 1000)



# Run the trained model on the entire test dataset
for epoch, (inp, tar) in enumerate(test_dataset):
    generate_images(generator, inp, tar, base_path='img/0', epoch=epoch)


