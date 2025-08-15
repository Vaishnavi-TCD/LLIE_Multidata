from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, ssim

from utils import save_img, VGGPerceptualLoss, torchPSNR
from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, rgb_to_y, ContrastLoss
from model import my_model
import kornia
import multiprocessing


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dirs, reference_dirs, transform=None, reference_transform=None):
        """
        Args:
            input_dirs (list): List of directories containing input images.
            reference_dirs (list): List of directories containing reference images.
            transform (callable, optional): Optional transform to be applied to input images.
            reference_transform (callable, optional): Optional transform to be applied to reference images.
        """
        self.input_dirs = input_dirs
        self.reference_dirs = reference_dirs
        self.transform = transform
        self.reference_transform = reference_transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        # Gather image paths from multiple directories
        self.input_images = []
        self.reference_images = []
        for input_dir, reference_dir in zip(input_dirs, reference_dirs):
            input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
            reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_extensions)])
           
            # Ensure filenames match, skip if not
            for file in input_files:
                if file in reference_files:
                    self.input_images.append(os.path.join(input_dir, file))
                    self.reference_images.append(os.path.join(reference_dir, file))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load images and apply transformations
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        reference_image = Image.open(self.reference_images[idx]).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
        if self.reference_transform:
            reference_image = self.reference_transform(reference_image)
        else:
            reference_image = self.transform(reference_image)  # Apply default transform if no separate one provided

        # Assuming input_image is a tensor of shape (3, H, W) in RGB format
        grayscale_image = 0.2989 * input_image[0] + 0.5870 * input_image[1] + 0.1140 * input_image[2]
        grayscale_image = grayscale_image.unsqueeze(0)  # Add channel dimension to get shape (1, H, W)


        return input_image, reference_image, idx

# Data loading functions for multiple directories
def get_dataset(root_dirs):
    """
    Args:
        root_dirs (dict): Dictionary containing dataset folder paths with 'input' and 'reference' keys.
                          Example: {'input': ['/path/to/uw_data', '/path/to/LUSI'], 
                                    'reference': ['/path/to/ref_data1', '/path/to/ref_data2']}
    """
    transform = transforms.Compose([
        transforms.Resize((256, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_dirs = root_dirs['input']
    reference_dirs = root_dirs['reference']
    return CustomImageDataset(input_dirs, reference_dirs, transform)

# Training settings
parser = argparse.ArgumentParser(description='Low-Light Image Enhancement Training')
parser.add_argument('--dataset', required=False, default='./LOL_data/', help='dataset folder')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--hight', type=int, default=256, help='height of images')
parser.add_argument('--width', type=int, default=256, help='width of images')
parser.add_argument('--finetune', default=False, help='to finetune')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=400, help='# of epochs at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of epochs to decay learning rate')
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy')
parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--edge_loss', default=False, help='apply edge loss for training')
parser.add_argument('--edge_loss_type', default='sobel', help='apply canny or sobel loss')


def main():
    opt = parser.parse_args()

    print(opt)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



    # if opt.cuda and not torch.cuda.is_available():
    #     raise Exception("No GPU found, please run without --cuda")

    # if the user asked for CUDA but there’s no GPU, just switch to CPU
    if opt.cuda and not torch.cuda.is_available():
        print("→ No GPU found; falling back to CPU")
        opt.cuda = False

    # ----------------------------------------------------------------
    # DataLoader kwargs: use multiple workers + pinning when on GPU
    loader_kwargs = {
        'num_workers': opt.threads,
        'pin_memory': True
    } if opt.cuda else {}
    # ----------------------------------------------------------------



    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    if opt.finetune:
        G_path = './best_checkpoint/netG_model_best_psnr_21.8953_ssim_0.9967.pth'
        net_g = torch.load(G_path).to(device)
    else:
        net_g = my_model().to(device)

    print(f'Trainable parameters: {count_parameters(net_g)}')    

    print('===> Loading datasets')
    root_dirs_train = {
        'input': ['./lol_dataset/our485/low', "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Low", "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Low"],
        'reference': ['./lol_dataset/our485/high', "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Normal", "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Normal"]
    }

    dataset_train = get_dataset(root_dirs_train)

    # data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **loader_kwargs)
    print(len(data_loader_train ))
    root_dirs_test = {
        'input': ['./lol_dataset/eval15/low', "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Low", "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Low"],
        'reference': ['./lol_dataset/eval15/high', "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Normal", "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Normal"]
    }
    dataset_test = get_dataset(root_dirs_test)

    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, **loader_kwargs)

    print('===> Building models')


    class Gradient_Loss(nn.Module):
        def __init__(self):
            super(Gradient_Loss, self).__init__()

            kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                        [[0,1,0],[1,-4,1],[0,1,0]],
                        [[0,1,0],[1,-4,1],[0,1,0]]]
            kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
            self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

        def forward(self, x,xx):
            grad = 0
            y = x
            yy = xx
            gradient_x = F.conv2d(y, self.weight_g,groups=3)
            gradient_xx = F.conv2d(yy,self.weight_g,groups=3)
            l = nn.L1Loss()
            a = l(gradient_x,gradient_xx)
            grad = grad + a
            return grad


    # Define losses
    Gradient_Loss = Gradient_Loss().to(device)
    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    L_per = VGGPerceptualLoss().to(device)
    MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
    Charbonnier_loss = nn.SmoothL1Loss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)

    # Initialize best PSNR and SSIM trackers
    # Initialize best PSNR and SSIM trackers
    best_psnr = 0
    best_ssim = 0

    # Initial weights for the loss functions
    weights = {
        'Charbonnier': 3.0,
        'Perceptual': 2.0,
        'MS_SSIM': 1.0,
        'Gradient': 2.5,
        'Edge': 0.5,  # Initially, edge loss is included conditionally
    }

    # Hyperparameters for tuning
    adjustment_factor = 0.1  # Factor by which to adjust weights
    improvement_threshold = 0.01  # Minimum improvement to adjust weights
    output_dir_train = './images_train'
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        net_g.train()
        for iteration, batch in enumerate(data_loader_train , 1):
            # rgb, tar, indx = batch[0].to(device), batch[1].to(device), batch[2]

            rgb   = batch[0].to(device, non_blocking=True)
            tar   = batch[1].to(device, non_blocking=True)
            indx  = batch[2]
            fake_b = net_g(rgb)
            # print(fake_b.shape,tar.shape)

            optimizer_g.zero_grad()
            
            # Calculate the total loss using adaptive weights
            loss_g_l1 = (weights['Charbonnier'] * Charbonnier_loss(fake_b, tar)
                        + weights['Perceptual'] * L_per(fake_b, tar)
                        + weights['MS_SSIM'] * (1 - MS_SSIM_loss(fake_b, tar))
                        + weights['Gradient'] * Gradient_Loss(fake_b, tar))

            loss_g = loss_g_l1
            
            if opt.edge_loss:
                if opt.edge_loss_type == 'canny':
                    edge_out1 = kornia.filters.canny(fake_b)
                    edge_gt = kornia.filters.canny(tar)
                    edge_loss = criterionL1(edge_out1[1], edge_gt[1])
                else:
                    # Apply Sobel correctly
                    fake_b_gray = kornia.color.rgb_to_grayscale(fake_b)  # Convert to grayscale
                    tar_gray = kornia.color.rgb_to_grayscale(tar)        # Convert to grayscale

                    edge_out1 = kornia.filters.sobel(fake_b_gray)  # Apply Sobel to the generated image
                    edge_gt = kornia.filters.sobel(tar_gray)       # Apply Sobel to the target image

                    edge_loss = criterionL1(edge_out1, edge_gt)  # Compare edges directly


                loss_g += weights['Edge'] * edge_loss

            loss_g.backward()
            optimizer_g.step()

            if iteration % 100 == 0:
                out_image = torch.cat((rgb, fake_b, tar), 3)
                save_img(out_image[0].detach().cpu(), f'images_train/{iteration}.png')
                print(f"===> Epoch[{epoch}]({iteration}/{len(data_loader_train)}): Loss_G: {loss_g.item()}")

        update_learning_rate(net_g_scheduler, optimizer_g)

        # Test and evaluate PSNR/SSIM for each epoch



        # Ensure output directory exists
        output_dir = './images_test'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        net_g.eval()
        avg_psnr = 0
        avg_ssim = 0
        for test_iter, batch in enumerate(data_loader_test, 1):
            # rgb_input, target, idx = batch[0].to(device), batch[1].to(device), batch[2]
            rgb_input, target, idx = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2]
            prediction = net_g(rgb_input)
            out = torch.cat((rgb_input, prediction, target), 3)

            # Now save the image
            filename = f'{output_dir}/{idx[0]}.png'  # Append the .png extension
            save_img(out[0].detach().cpu(), filename)

            psnr = torchPSNR(prediction, target).item()
            avg_psnr += psnr
            avg_ssim += ssim(prediction, target).item()

        avg_psnr /= len(data_loader_test)
        avg_ssim /= len(data_loader_test)

        print(f"===> Avg. PSNR: {avg_psnr:.4f} dB, Avg. SSIM: {avg_ssim:.4f}")

        # Save the best model based on PSNR and SSIM
        if avg_psnr > best_psnr or avg_ssim > best_ssim:
            best_psnr = max(best_psnr, avg_psnr)
            best_ssim = max(best_ssim, avg_ssim)
            best_model_out_path = f"best_checkpoint/netG_model_best_mixed_psnr_{best_psnr:.4f}_ssim_{best_ssim:.4f}.pth"
            if not os.path.exists("best_checkpoint"):
                os.mkdir("best_checkpoint")
            torch.save(net_g, best_model_out_path)
            print(f"Best model saved at {best_model_out_path}")

        # Regular checkpoint saving
        net_g_model_out_path = f"checkpoint/{opt.dataset}/netG_model_mixed_epoch_{epoch}_psnr_{avg_psnr:.4f}.pth"
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        torch.save(net_g, net_g_model_out_path)
        print(f"Checkpoint saved at {net_g_model_out_path}")

        # Adaptive weight adjustment based on validation metrics
        if avg_psnr > best_psnr + 0.01:  # Adjust if there's a significant improvement
            weights['Charbonnier'] += adjustment_factor
        else:
            weights['Charbonnier'] -= adjustment_factor if weights['Charbonnier'] > 0 else 0

        if avg_ssim > best_ssim + 0.01:  # Adjust if there's a significant improvement
            weights['MS_SSIM'] += adjustment_factor
        else:
            weights['MS_SSIM'] -= adjustment_factor if weights['MS_SSIM'] > 0 else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
