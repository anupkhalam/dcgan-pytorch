#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:06:57 2022.

@author: anup
"""

import os
import gc
import time
import torch
import random
import shutil
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import HTML
import torchvision.utils as vutils
import torchvision.datasets as dset
from collections import defaultdict
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.animation as animation
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

try:
    shutil.rmtree(os.path.join('runs', 'PitchClass'))
    shutil.rmtree(os.path.join('log', 'vgg16'))
except FileNotFoundError:
    pass
writer = SummaryWriter('runs/PitchClass')


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results


def seed_all(SEED_VALUE=999):
    """Set the seed for consistency."""
    random.seed(SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return None


def start_timer():
    """Record start time."""
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()
    return None


def end_timer_and_print(local_msg):
    """Record end time."""
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(
        torch.cuda.max_memory_allocated()))
    return None


def get_transform(image_size):
    """Get transformation for train data."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])


def get_data_loader(dataset, params, shuffle_switch):
    """Define the datalodader for the validation dataset."""
    return DataLoader(
                        dataset,
                        batch_size=params["batch_size"],
                        shuffle=shuffle_switch,
                        num_workers=params["workers"],
                        pin_memory=True,
                        drop_last=True,
                    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return None


class Generator(nn.Module):
    def __init__(self, ngpu, nc, ngf, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )


    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, nz):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class MetricMonitor:
    """Class for defining metrics."""

    def __init__(self, float_precision=3):
        """Receive the precision strngth."""
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        """Define metric."""
        self.metrics = defaultdict(lambda: {"errD": np.array([]),
                                            "errG": np.array([]),
                                            "D_x": np.array([]),
                                            "D_G_z1": np.array([]),
                                            "D_G_z2": np.array([])})


    def extend(self, epoch, item, val):
        """Define metric."""
        metric = self.metrics[epoch]
        metric[item] = np.append(metric[item], val)

    def get_epoch_report(self, epoch):
        """Define metric."""
        return {"epoch_average_errD":
                np.average(self.metrics[epoch]["errD"]),
                "epoch_average_errG":
                np.average(self.metrics[epoch]["errG"]),
                "epoch_average_D_x":
                np.average(self.metrics[epoch]["D_x"]),
                "epoch_average_D_G_z1":
                np.average(self.metrics[epoch]["D_G_z1"]),
                "epoch_average_D_G_z2":
                np.average(self.metrics[epoch]["D_G_z2"]),
                }



def train(epoch,
          netD,
          netG,
          criterion,
          params,
          optimizerD,
          optimizerG,
          dataloader):
    """Train data."""
    netD.train()
    netG.train()
    real_label = 1.
    fake_label = 0.
    metric_monitor = MetricMonitor()

    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(params['device'])
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),
                           real_label,
                           dtype=torch.float,
                           device=params['device'])
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, params['nz'], 1, 1, device=params['device'])
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print("="*100)
            print(
                  f"Epoch Status         : {epoch} / {params['num_epochs']}"
                 )
            print(f"DataLoader Status    : {i} / {len(dataloader)}")
            print(f"Discriminator Error  : {errD.item()}")
            print(f"Generator Error      : {errG.item()}")
            print(f"D_x                  : {D_x}")
            print(f"D_G_z1               : {D_G_z1}")
            print(f"D_G_z2               : {D_G_z2}")

        # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == params['num_epochs']-1) and
        #                           (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake,
        #                                      padding=2,
        #                                      normalize=True))
        metric_monitor.extend(epoch, "errD", errD.item())
        metric_monitor.extend(epoch, "errG", errG.item())
        metric_monitor.extend(epoch, "D_x", D_x)
        metric_monitor.extend(epoch, "D_G_z1", D_G_z1)
        metric_monitor.extend(epoch, "D_G_z2", D_G_z2)
    epoch_out = {}
    epoch_out['metric_monitor'] = metric_monitor
    epoch_out['modelD'] = netD
    epoch_out['modelG'] = netG
    epoch_out['optimizerD'] = optimizerD
    epoch_out['optimizerG'] = optimizerG
    epoch_out['epoch'] = epoch
    return epoch_out


def save_checkpoint(state, filename = "model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at ", filename)

def load_checkpoint(checkpoint, netD, netG, optimizerD, optimizerG):
    print("Loading checkpoint...")
    netD.load_state_dict(checkpoint['state_dictD'])
    netG.load_state_dict(checkpoint['state_dictG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    last_epoch = checkpoint['epoch']
    last_errD = checkpoint['errD']
    last_errG = checkpoint['errG']
    return netD, netG, optimizerD, optimizerG, last_epoch, last_errD, last_errG


def main():
    """Process control function."""
    seed_all(SEED_VALUE=999)

    params = {
                'dataroot': "/home/anup/01_work/02_self/GAN/DCGAN/data/CelebA-20220927T094243Z-001/CelebA/Img",
                'workers': 6,
                'batch_size': 1024,
                'image_size': 64,
                'nc': 3,
                'nz': 100,
                'ngf': 64,
                'ndf': 64,
                'num_epochs': 16,
                'lr': 0.0002,
                'beta1': 0.5,
                'ngpu': 1,
                'model_path': os.path.join('models', 'model_scripted.pt'),
                'checkpoint_name': os.path.join('checkpoints',
                                                'model_checkpoint.pth.tar'),
             }
    device = torch.device("cuda:0" if (torch.cuda.is_available() and
                                       params['ngpu'] > 0) else "cpu")
    params['device'] = device
    dataset = dset.ImageFolder(root=params['dataroot'],
                               transform=get_transform(params['image_size']))
    dataloader = get_data_loader(dataset,
                                 params=params,
                                 shuffle_switch=True)
    netG = Generator(params['ngpu'],
                     params['nc'],
                     params['ngf'],
                     params['nz'],
                     ).to(device)
    if (device.type == 'cuda') and (params['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(params['ngpu'])))
    netD = Discriminator(params['ngpu'],
                         params['nc'],
                         params['ndf'],
                         params['nz'],
                         ).to(device)

    if (device.type == 'cuda') and (params['ngpu'] > 1):
        netD = nn.DataParallel(netD, list(range(params['ngpu'])))
    if os.path.exists(params['checkpoint_name']):
        optimizerD = optim.Adam(netD.parameters(),
                                lr=params['lr'],
                                betas=(params['beta1'], 0.999))
        optimizerG = optim.Adam(netG.parameters(),
                                lr=params['lr'],
                                betas=(params['beta1'], 0.999))
        netD, netG, optimizerD, optimizerG, last_epoch, last_errD, last_errG =\
            load_checkpoint(
                torch.load(
                    params['checkpoint_name']),
                netD, netG, optimizerD, optimizerG)

        print("Loading Checkpoint.")
        print("Epoch Loaded : ", last_epoch)
        print("Last errD : ", last_errD)
        print("Last errG : ", last_errG)
    else:
        print('Checkpoint do not exist.')
        last_epoch = 0
        last_errD = np.inf
        last_errG = np.inf
        netD.apply(weights_init)
        netG.apply(weights_init)

        optimizerD = optim.Adam(netD.parameters(),
                                lr=params['lr'],
                                betas=(params['beta1'], 0.999))
        optimizerG = optim.Adam(netG.parameters(),
                                lr=params['lr'],
                                betas=(params['beta1'], 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
    print("Starting Training Loop...")
    for iters, epoch in enumerate(range(last_epoch + 1, params['num_epochs'])):
        epoch_out = train(epoch,
                          netD,
                          netG,
                          criterion,
                          params,
                          optimizerD,
                          optimizerG,
                          dataloader)
        netD = epoch_out['modelD']
        netG = epoch_out['modelG']
        epoch_out_metric = epoch_out['metric_monitor'].get_epoch_report(epoch)

        if last_errD > epoch_out_metric['epoch_average_errD'] and \
           last_errG > epoch_out_metric['epoch_average_errG']:

            checkpoint = {'epoch': epoch_out['epoch'],
                          'state_dictD': epoch_out['modelD'].state_dict(),
                          'state_dictG': epoch_out['modelG'].state_dict(),
                          'optimizerD': epoch_out['optimizerD'].state_dict(),
                          'optimizerG': epoch_out['optimizerG'].state_dict(),
                          'errD': epoch_out_metric['epoch_average_errD'],
                          'errG': epoch_out_metric['epoch_average_errG'],
                          'D_x': epoch_out_metric['epoch_average_D_x'],
                          'D_G_z1': epoch_out_metric['epoch_average_D_G_z1'],
                          'D_G_z2': epoch_out_metric['epoch_average_D_G_z2']}
            print("Saving checkpoint.")
            save_checkpoint(checkpoint, params['checkpoint_name'])
            last_errD = epoch_out_metric['epoch_average_errD']
            last_errG = epoch_out_metric['epoch_average_errG']
    model_scripted = torch.jit.script(netG)
    model_scripted.save('models/model_scripted.pt')

if __name__ == "__main__":
    main()


    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses, label="G")
    # plt.plot(D_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


    # fig = plt.figure(figsize=(8,8))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # HTML(ani.to_jshtml())

    # # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))

    # # Plot the real images
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # # Plot the fake images from the last epoch
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # plt.show()



# c = torch.randn(b_size, nz, 1, 1, device=device)
# # Generate fake image batch with G
# f = netG(c).detach().cpu()
# f = netG(c).detach().cpu()[0].squeeze().permute(1,2,0)
# f = vutils.make_grid(f, padding=2, normalize=True).squeeze().permute(1,2,0)
# f = f.numpy()
# f.shape

# from matplotlib import pyplot as plt
# plt.imshow(f, interpolation='nearest')
# plt.show()





