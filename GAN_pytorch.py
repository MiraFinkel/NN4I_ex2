import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.style.use('ggplot')

# ====================== learning parameters ====================== #
batch_size = 512
epochs = 200
fixed_sample_size = 64
latent_vector_size = 128
steps_num_discriminator = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
to_pil_image = transforms.ToPILImage()

train_data = datasets.MNIST(root='../input/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.model = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.model = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)


def label_real(size):
    """
    Creates real labels (1s)
    :param size:
    :return:
    """
    data = torch.ones(size, 1)
    return data.to(device)


def label_fake(size):
    """
    Creates fake labels (0s)
    :param size:
    :return:
    """
    data = torch.zeros(size, 1)
    return data.to(device)


def create_noise_vector(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)


def save_generator_image(image, path):
    save_image(image, path)


def train_discriminator(discriminator, generator, optimizer, data_real, data_fake, criterion, epoch):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), '../outputs/discriminator_epoch' + str(epoch) + '.pth')
    return loss_real + loss_fake


def train_generator(discriminator, optimizer, data_fake, criterion):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label)
    loss.backward()
    optimizer.step()
    return loss


def print_models(generator, discriminator):
    print('##### GENERATOR #####')
    print(generator)
    print('######################')
    print('\n##### DISCRIMINATOR #####')
    print(discriminator)
    print('######################')


def get_optimizers(generator, discriminator):
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    return generator_optimizer, discriminator_optimizer


def train_by_batch(criterion, data, discriminator, discriminator_optimizer, epoch, generator, generator_optimizer,
                   loss_discriminator, loss_generator):
    image, _ = data
    image = image.to(device)
    b_size = len(image)
    for step in range(steps_num_discriminator):  # run the discriminator for k steps
        data_fake = generator(create_noise_vector(b_size, latent_vector_size)).detach()
        data_real = image
        loss_discriminator += train_discriminator(discriminator, generator, discriminator_optimizer, data_real,
                                                  data_fake,
                                                  criterion, epoch)
    data_fake = generator(create_noise_vector(b_size, latent_vector_size))
    loss_generator += train_generator(discriminator, generator_optimizer, data_fake, criterion)
    return loss_discriminator, loss_generator


def main():
    generator = Generator(latent_vector_size).to(device)
    discriminator = Discriminator().to(device)

    generator_optimizer, discriminator_optimizer = get_optimizers(generator, discriminator)

    criterion = nn.BCELoss()  # loss function

    losses_g, losses_d, images = [], [], []

    noise = create_noise_vector(fixed_sample_size, latent_vector_size)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        loss_generator = 0.0
        loss_discriminator = 0.0
        num = 0
        for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
            loss_discriminator, loss_generator = train_by_batch(criterion, data, discriminator, discriminator_optimizer,
                                                                epoch, generator,
                                                                generator_optimizer, loss_discriminator, loss_generator)
            num = bi
        generated_img = generator(noise).cpu().detach()  # create fake image
        generated_img = make_grid(generated_img)
        save_generator_image(generated_img, f"../outputs/gen_img{epoch}.png")
        images.append(generated_img)
        epoch_loss_g = loss_generator / num  # total generator loss
        epoch_loss_d = loss_discriminator / num  # total discriminator loss
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)

        print(f"Epoch {epoch} of {epochs}")
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), '../outputs/generator_epoch' + str(epoch) + '.pth')

    print('DONE TRAINING')
    torch.save(generator.state_dict(), '../outputs/generator.pth')

    # save the generated images as GIF file
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('../outputs/generator_images.gif', imgs)

    # plot and save the generator and discriminator loss
    plt.figure()
    plt.plot(losses_g, label='Generator loss')
    plt.plot(losses_d, label='Discriminator Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')


if __name__ == '__main__':
    main()

# optimizers
