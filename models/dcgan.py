import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from utils.tensorboard_logger import Logger
from utils.inception_score import get_inception_score
from itertools import chain
from torchvision import utils

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        self.cuda = False
        # check if cuda is available
        self.check_cuda(args.cuda)
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        self.G.to(self.device)
        self.D.to(self.device)
        self.loss.to(self.device)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Set the logger
        self.logger = Logger('./logs')
        self.number_of_images = 10

    # cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag and torch.cuda.is_available():
            self.cuda = True
        print("Cuda enabled flag: ", self.cuda)

    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                images = images.view(self.batch_size, -1).to(self.device)
                z = torch.randn(self.batch_size, 100, 1, 1, device=self.device)
                real_labels = torch.ones(self.batch_size, device=self.device)
                fake_labels = torch.zeros(self.batch_size, device=self.device)

                # Train discriminator
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.flatten(), real_labels)

                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)

                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                z = torch.randn(self.batch_size, 100, 1, 1, device=self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs.flatten(), real_labels)

                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if generator_iter % 1000 == 0:
                    # Workaround because graphic card memory can't store more than 800+ examples in memory for generating image
                    # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                    # This way Inception score is more correct since there are different generated examples from every class of Inception model
                    # sample_list = []
                    # for i in range(10):
                    #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                    #     samples = self.G(z)
                    #     sample_list.append(samples.data.cpu().numpy())
                    #
                    # # Flattening list of lists into one list of numpy arrays
                    # new_sample_list = list(chain.from_iterable(sample_list))
                    # print("Calculating Inception Score over 8k generated images")
                    # # Feeding list of numpy arrays
                    # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                    #                                       resize=True, splits=10)
                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()
                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')
                    z_samples = torch.randn(800, 100, 1, 1, device=self.device)
                    samples = self.G(z_samples)
                    samples = samples.mul(0.5).add(0.5).cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(grid, f'training_result_images/img_generatori_iter_{str(generator_iter).zfill(3)}.png')

                if ((i + 1) % 100) == 0:
                    print(f"Epoch: [{epoch+1}] [{i+1}/{train_loader.dataset.__len__() // self.batch_size}] D_loss: {d_loss.item():.8f}, G_loss: {g_loss.item():.8f}")

                    z_summary = torch.randn(self.batch_size, 100, 1, 1, device=self.device)
                    info = {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, generator_iter)
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value, generator_iter)
                        self.logger.histo_summary(tag + '/grad', value.grad, generator_iter)
                    info_images = {
                        'real_images': self.real_images(images, self.number_of_images),
                        'generated_images': self.generate_img(z_summary, self.number_of_images)
                    }
                    for tag, images in info_images.items():
                        self.logger.image_summary(tag, images, generator_iter)

        self.t_end = t.time()
        print(f'Time of training-{(self.t_end - self.t_begin)}')
        self.logger.close()
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        self.G.eval()
        self.D.eval()
        z = torch.randn(self.batch_size, 100, 1, 1, device=self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5).cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'dgan_model_image.png')
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")

    def load_model(self, D_model_filename, G_model_filename):
        if D_model_filename is not None and os.path.exists(D_model_filename):
            self.D.load_state_dict(torch.load(D_model_filename))
            print(f"Discriminator loaded from {D_model_filename}")
        else:
            print("No Discriminator model provided, skipping loading.")

        if G_model_filename is not None and os.path.exists(G_model_filename):
            self.G.load_state_dict(torch.load(G_model_filename))
            print(f"Generator loaded from {G_model_filename}")
        else:
            print("No Generator model provided, skipping loading.")

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1, device=self.device)
        z1 = torch.randn(1, 100, 1, 1, device=self.device)
        z2 = torch.randn(1, 100, 1, 1, device=self.device)

        images = []
        alpha = 1.0 / float(number_int + 1)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5).cpu()
            images.append(fake_im.view(self.C, 32, 32))

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, f'interpolated_images/interpolated_{str(number).zfill(3)}.png')
        print(f"Saved interpolated images to interpolated_images/interpolated_{str(number).zfill(3)}.")

