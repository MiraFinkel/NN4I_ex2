# import keras
# from keras import layers
# from keras.datasets import mnist
# import numpy as np
# from keras import regularizers
# import matplotlib.pyplot as plt
# from keras.callbacks import TensorBoard
#
# # This is the size of our encoded representations
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
#
#
# def get_data_and_reprocess():
#     (x_train, _), (x_test, _) = mnist.load_data()
#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.
#     # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#     # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#     x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
#     x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#     print(x_train.shape)
#     print(x_test.shape)
#     return x_test, x_train
#
#
# def simple_encoder_decoder(x_test, x_train, epochs_num=50):
#     # This is our input image
#     input_img = keras.Input(shape=(784,))
#     # "encoded" is the encoded representation of the input
#     encoded = layers.Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
#     # "decoded" is the lossy reconstruction of the input
#     decoded = layers.Dense(784, activation='sigmoid')(encoded)
#     # This model maps an input to its reconstruction
#     autoencoder = keras.Model(input_img, decoded)
#     # This model maps an input to its encoded representation
#     encoder = keras.Model(input_img, encoded)
#     # This is our encoded (32-dimensional) input
#     encoded_input = keras.Input(shape=(encoding_dim,))
#     # Retrieve the last layer of the autoencoder model
#     decoder_layer = autoencoder.layers[-1]
#     # Create the decoder model
#     decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     autoencoder.fit(x_train, x_train, epochs=epochs_num, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
#     # Encode and decode some digits
#     # Note that we take them from the *test* set
#     encoded_imgs = encoder.predict(x_test)
#     decoded_imgs = decoder.predict(encoded_imgs)
#     return decoded_imgs
#
#
# def convolutional_autoencoder():
#     input_img = keras.Input(shape=(28, 28, 1))
#
#     x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     x = layers.Dense(activation='relu', padding='same')(x)
#     x = layers.Dense(8, (3, 3), activation='relu', padding='same')(x)
#     encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
#
#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional
#
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(16, (3, 3), activation='relu')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
#     autoencoder = keras.Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     return autoencoder
#
#
# def plot_images(x_test, decoded_imgs):
#     n = 10  # How many digits we will display
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # Display original
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(x_test[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         # Display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(decoded_imgs[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#
#
# def run_simple_encoder_decoder(epochs_num=50):
#     x_test, x_train = get_data_and_reprocess()
#     decoded_imgs = simple_encoder_decoder(x_test, x_train, epochs_num)
#     plot_images(x_test, decoded_imgs)
#
#
# def run_convolutional_autoencoder():
#     x_test, x_train = get_data_and_reprocess()
#     autoencoder = convolutional_autoencoder()
#     autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
#                     callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
#     decoded_imgs = autoencoder.predict(x_test)
#     plot_images(x_test, decoded_imgs)
#
#
# if __name__ == '__main__':
#     # run_simple_encoder_decoder(epochs_num=100)
#     run_convolutional_autoencoder()
#     a = 7















# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# #Converting data to torch.FloatTensor
# transform = transforms.ToTensor()
#
# # Download the training and test datasets
# train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
#
# test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
#
# #Prepare data loaders
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)
#
#
# #Utility functions to un-normalize and display an image
# def imshow(img):
#     img = img / 2 + 0.5
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#
#
# #Define the image classes
# classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
#            '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
#
# #Obtain one batch of training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# images = images.numpy() # convert images to numpy for display
#
# #Plot the images
# fig = plt.figure(figsize=(8, 8))
# # display 20 images
# for idx in np.arange(9):
#     ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(classes[labels[idx]])
#
#
#
# #Define the Convolutional Autoencoder
# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#
#         #Encoder
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         #Decoder
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
#
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.t_conv1(x))
#         x = F.sigmoid(self.t_conv2(x))
#
#         return x
#
#
# #Instantiate the model
# model = ConvAutoencoder()
# print(model)
#
#
#
# #Loss function
# criterion = nn.BCELoss()
#
# #Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
#
# def get_device():
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     return device
#
# device = get_device()
# print(device)
# model.to(device)
#
#
#
# #Epochs
# n_epochs = 100
#
# for epoch in range(1, n_epochs+1):
#     # monitor training loss
#     train_loss = 0.0
#
#     #Training
#     for data in train_loader:
#         images, _ = data
#         images = images.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, images)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*images.size(0)
#
#     train_loss = train_loss/len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
#
#
#     #Batch of test images
# dataiter = iter(test_loader)
# images, labels = dataiter.next()
# batch_size = 32
#
# #Sample outputs
# output = model(images)
# images = images.numpy()
#
# output = output.view(batch_size, 3, 32, 32)
# output = output.detach().numpy()
#
# #Original Images
# print("Original Images")
# fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
# for idx in np.arange(5):
#     ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(classes[labels[idx]])
# plt.show()
#
# #Reconstructed Images
# print('Reconstructed Images')
# fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
# for idx in np.arange(5):
#     ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
#     imshow(output[idx])
#     ax.set_title(classes[labels[idx]])
# plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)[:4096]


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--
    train_loader = torch.utils.data.DataLoader(mnist_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs
def interpolate(index1, index2):
    x1 = mnist_data[index1][0]
    x2 = mnist_data[index2][0]
    x = torch.stack([x1, x2])
    embedding = model.encoder(x)
    e1 = embedding[0]  # embedding of first image
    e2 = embedding[1]  # embedding of second image


    embedding_values = []
    for i in range(0, 10):
        e = e1 * (i/10) + e2 * (10-i)/10
        embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model.decoder(embedding_values)

    plt.figure(figsize=(10, 2))
    for i, recon in enumerate(recons.detach().numpy()):
        plt.subplot(2, 10, i+1)
        plt.imshow(recon[0])
    plt.subplot(2, 10, 11)
    plt.imshow(x2[0])
    plt.subplot(2, 10, 20)
    plt.imshow(x1[0])

model = Autoencoder()
max_epochs = 20
outputs = train(model, num_epochs=max_epochs)