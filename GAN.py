import matplotlib.pyplot as plt
import numpy as np
from keras.datasets.mnist import load_data
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model


def get_discriminator_model(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))  # first convolution
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))   # second convolution
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_generator_model(latent_dim):
    model = Sequential()
    # foundation for 7x7 image - MNIST
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # up-sample to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # up-sample to 28x28
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
    return model


def generate_fake_samples_discriminator(n_samples):
    # generate uniform random numbers in [0,1]
    X = np.random.rand(28 * 28 * n_samples)
    # reshape into a batch of grayscale images
    X = X.reshape((n_samples, 28, 28, 1))
    # generate 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_iter):
        # get randomly selected 'real' samples
        x_real, y_real = generate_samples(dataset, half_batch)
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(x_real, y_real)
        # generate 'fake' examples
        x_fake, y_fake = generate_fake_samples_discriminator(half_batch)
        # update discriminator on fake samples
        _, fake_acc = model.train_on_batch(x_fake, y_fake)
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y

    # generate points in latent space as input for the generator


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def get_data():
    # load the images into memory
    (train_x, train_y), (test_x, test_y) = load_data()
    # summarize the shape of the dataset
    print('Train', train_x.shape, train_y.shape)
    print('Test', test_x.shape, test_y.shape)
    x = np.expand_dims(train_x, axis=-1)
    # convert from unsigned ints to floats
    x = x.astype('float32')
    # scale from [0,255] to [0,1]
    x = x / 255.0
    return x, train_x, train_y, test_x, test_y


def generate_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def save_model(model):
    # summarize the model
    model.summary()
    # plot the model
    plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)


def run_discriminator_training(discriminator):
    # define the discriminator model
    model = discriminator.get_model()
    # load image data
    dataset, _, _, _, _ = get_data()
    # fit the model
    discriminator.train(model, dataset)


def show_fake_samples_generator():
    # size of the latent space
    latent_dim = 100
    # define the discriminator model
    model = get_generator_model(latent_dim)
    # generate samples
    n_samples = 25
    X, _ = generate_fake_samples(model, latent_dim, n_samples)
    # plot the generated samples
    for i in range(n_samples):
        # define subplot
        plt.subplot(5, 5, 1 + i)
        # turn off axis labels
        plt.axis('off')
        # plot single image
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    # show the figure
    plt.show()


# define the combined generator and discriminator model, for updating the generator
def get_gan_model(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=256):
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            x_real, y_real = generate_samples(dataset, half_batch)
            # generate 'fake' examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(x, y)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
            # evaluate the model performance, sometimes
            if (i + 1) % 10 == 0:
                summarize_performance(i, g_model, d_model, dataset, latent_dim)


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    x_real, y_real = generate_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    data, _, _, _, _ = get_data()
    # size of the latent space
    cur_latent_dim = 100
    # create the discriminator
    discriminator_model = get_discriminator_model()
    # create the generator
    generator_model = get_generator_model(cur_latent_dim)
    # create the gan
    cur_gan_model = get_gan_model(generator_model, discriminator_model)
    # summarize gan model
    cur_gan_model.summary()

    train(generator_model, discriminator_model, cur_gan_model, data, cur_latent_dim)
