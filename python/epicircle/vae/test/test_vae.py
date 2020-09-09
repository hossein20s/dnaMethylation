import unittest

import numpy
import tensorflow
from tensorflow.python.training.checkpoint_management import latest_checkpoint
from tensorflow.python.training.tracking.util import Checkpoint
from matplotlib import pyplot as plt

from epicircle.vae.mnist_processor import plot_latent_images, display_image, generate_and_save_images, MnistProcessor
from epicircle.vae.vae import Conv_VAE

LATENT_DEPTH = 20

BATCH_SIZE = 32
NUM_EPOCHS = 100

MODEL_SAVE_DIR = "../../learned-models/VAE-MNIST-tf2"
MODEL_NAME = "VAE-MNIST"

DATASET_PATH = "../../datasets/mnist-in-csv/"


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.epochs = 10
        self.latent_dim = 2
        self.batch_size = 32
        self.processor = MnistProcessor(train_size=60000, batch_size=self.batch_size, test_size=10000)

    def test_vae_decoder(self):
        # set the dimensionality of the latent space to a plane for visualization later
        num_examples_to_generate = 16

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tensorflow.random.normal(
            shape=[num_examples_to_generate, self.latent_dim])
        model = Conv_VAE(self.latent_dim)
        self.processor.train(model=model, epochs=self.epochs)
        # generate_and_save_images(model, epoch, test_sample)

        # Pick a sample of the test set for generating output images
        assert self.batch_size >= num_examples_to_generate
        for test_batch in self.processor.test_dataset.take(1):
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    def test_decode(self):
        model = Conv_VAE(self.latent_dim)
        eps = numpy.random.normal(size=[150, LATENT_DEPTH]).astype(numpy.float32)
        ckpt = Checkpoint(encoder=model.encoder, decoder=model.decoder)
        ckpt.restore(latest_checkpoint(MODEL_SAVE_DIR))
        f = model.decode(eps, training=False)

    def test_plot_digits(self):
        plt.imshow(display_image(epoch))
        plt.axis('off')  # Display images
        plot_latent_images(model, 20)
        generate_and_save_images(model, 0, test_sample)
        # Pick a sample of the test set for generating output images
        assert batch_size >= num_examples_to_generate
        for test_batch in test_dataset.take(1):
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]
