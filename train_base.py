# train_base.py
from __future__ import print_function
import argparse
import os
from math import log10
import time
import csv
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import MS_SSIM, ssim
import kornia

# === Your existing project imports ===
from utils import save_img, VGGPerceptualLoss, torchPSNR
from network1 import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, rgb_to_y, ContrastLoss

# IMPORTANT: baseline model (no ablation changes)
from model import my_model


# ======= Helpers for CSV logging & runtime metrics =======
def count_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_latency_ms(model, device, h, w, warmup=10, iters=100):
    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / iters

def append_csv_row(csv_path, row_dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ======= Minimal dataset wrapper (same logic as your trainer) =======
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dirs, reference_dirs, transform=None, reference_transform=None):
        self.input_dirs = input_dirs
        self.reference_dirs = reference_dirs
        self.transform = transform
        self.reference_transform = reference_transform
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        self.input_images, self.reference_images = [], []
        for inp, ref in zip(input_dirs, reference_dirs):
            if not (os.path.isdir(inp) and os.path.isdir(ref)):
                continue
            input_files = sorted([f for f in os.listdir(inp) if f.lower().endswith(valid_ext)])
            reference_files = sorted([f for f in os.listdir(ref) if f.lower().endswith(valid_ext)])
            for file in input_files:
                if file in reference_files:
                    self.input_images.append(os.path.join(inp, file))
                    self.reference_images.append(os.path.join(ref, file))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert("RGB")
        reference_image = Image.open(self.reference_images[idx]).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
        if self.reference_transform:
            reference_image = self.reference_transform(reference_image)
        else:
            reference_image = self.transform(reference_image)

        return input_image, reference_image, idx


def get_dataset(root_dirs, height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return CustomImageDataset(root_dirs['input'], root_dirs['reference'], transform)


# ======= Args =======
parser = argparse.ArgumentParser(description='Low-Light Image Enhancement Training (Baseline w/ CSV logging + per-run outputs)')
parser.add_argument('--dataset', required=False, default='./LOL_data/', help='dataset folder (for checkpoint pathing)')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--hight', type=int, default=256, help='height of images')   # keeping your arg name
parser.add_argument('--width', type=int, default=384, help='width of images')
parser.add_argument('--finetune', default=False, help='to finetune from a checkpoint')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=60, help='# of epochs at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=40, help='# of epochs to decay learning rate')
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

# NEW: logging helpers
parser.add_argument('--run_tag', type=str, default='B0', help='Identifier for this baseline run')
parser.add_argument('--csv_out', type=str, default='ablation_results.csv', help='CSV file to append metrics')


def main():
    opt = parser.parse_args()
    print(opt)

    # --- Per-run output directories (Option 2) ---
    outdir    = os.path.join('outputs', opt.run_tag)
    img_train = os.path.join(outdir, 'images_train')
    img_test  = os.path.join(outdir, 'images_test')
    ckpt_best = os.path.join(outdir, 'best_checkpoint')
    ckpt_dir  = os.path.join(outdir, 'checkpoint', opt.dataset)
    for d in [img_train, img_test, ckpt_best, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if opt.cuda and not torch.cuda.is_available():
        print("→ No GPU found; falling back to CPU")
        opt.cuda = False

    loader_kwargs = {'num_workers': opt.threads, 'pin_memory': True} if opt.cuda else {}

    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    # ===== Model (Baseline) =====
    if opt.finetune:
        # NOTE: adjust this path if you want to finetune from a specific per-run best model
        G_path = os.path.join(ckpt_best, 'netG_model_best.psnr_ssim.pth')
        if not os.path.isfile(G_path):
            raise FileNotFoundError(f'--finetune was set, but file not found: {G_path}')
        net_g = torch.load(G_path).to(device)
    else:
        net_g = my_model().to(device)

    print(f'Trainable parameters: {count_parameters(net_g)}')

    # Static metrics (once per run)
    params_m = count_params_m(net_g)
    try:
        latency_ms = measure_latency_ms(net_g, device, opt.hight, opt.width)
    except Exception as e:
        print(f"[warn] latency measurement failed: {e}")
        latency_ms = None

    # ===== Data =====
    print('===> Loading datasets')
    root_dirs_train = {
        'input': ['./lol_dataset/our485/low',
                  "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Low",
                  "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Low"],
        'reference': ['./lol_dataset/our485/high',
                      "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Normal",
                      "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Normal"]
    }
    dataset_train = get_dataset(root_dirs_train, opt.hight, opt.width)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                    shuffle=True, **loader_kwargs)
    print(len(data_loader_train))

    root_dirs_test = {
        'input': ['./lol_dataset/eval15/low',
                  "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Low",
                  "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Low"],
        'reference': ['./lol_dataset/eval15/high',
                      "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Normal",
                      "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Normal"]
    }
    dataset_test = get_dataset(root_dirs_test, opt.hight, opt.width)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=True, **loader_kwargs)

    # ===== Losses/Optim =====
    print('===> Building losses/optim')
    class Gradient_Loss(nn.Module):
        def __init__(self):
            super(Gradient_Loss, self).__init__()
            kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                        [[0,1,0],[1,-4,1],[0,1,0]],
                        [[0,1,0],[1,-4,1],[0,1,0]]]
            kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
            self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

        def forward(self, x, xx):
            y, yy = x, xx
            gradient_x = F.conv2d(y, self.weight_g, groups=3)
            gradient_xx = F.conv2d(yy, self.weight_g, groups=3)
            return nn.L1Loss()(gradient_x, gradient_xx)

    Gradient_Loss = Gradient_Loss().to(device)
    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    L_per = VGGPerceptualLoss().to(device)
    MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
    Charbonnier_loss = nn.SmoothL1Loss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)

    best_psnr = 0
    best_ssim = 0

    # Keeping your default weights (unchanged)
    weights = {
        'Charbonnier': 3.0,
        'Perceptual': 2.0,
        'MS_SSIM': 1.0,
        'Gradient': 2.5,
        'Edge': 0.5,
    }

    print('===> Training')
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        net_g.train()
        for iteration, batch in enumerate(data_loader_train, 1):
            rgb   = batch[0].to(device, non_blocking=True)
            tar   = batch[1].to(device, non_blocking=True)
            _idx  = batch[2]

            fake_b = net_g(rgb)

            optimizer_g.zero_grad()
            loss_g = (weights['Charbonnier'] * Charbonnier_loss(fake_b, tar)
                      + weights['Perceptual'] * L_per(fake_b, tar)
                      + weights['MS_SSIM'] * (1 - MS_SSIM_loss(fake_b, tar))
                      + weights['Gradient'] * Gradient_Loss(fake_b, tar))

            if opt.edge_loss:
                if opt.edge_loss_type == 'canny':
                    edge_out1 = kornia.filters.canny(fake_b)
                    edge_gt = kornia.filters.canny(tar)
                    edge_loss = criterionL1(edge_out1[1], edge_gt[1])
                else:
                    fake_b_gray = kornia.color.rgb_to_grayscale(fake_b)
                    tar_gray = kornia.color.rgb_to_grayscale(tar)
                    edge_out1 = kornia.filters.sobel(fake_b_gray)
                    edge_gt = kornia.filters.sobel(tar_gray)
                    edge_loss = criterionL1(edge_out1, edge_gt)
                loss_g += weights['Edge'] * edge_loss

            loss_g.backward()
            optimizer_g.step()

            if iteration % 100 == 0:
                out_image = torch.cat((rgb, fake_b, tar), 3)
                save_img(out_image[0].detach().cpu(), os.path.join(img_train, f'{iteration}.png'))
                print(f"===> Epoch[{epoch}]({iteration}/{len(data_loader_train)}): Loss_G: {loss_g.item():.4f}")

        update_learning_rate(net_g_scheduler, optimizer_g)

        # ===== Eval =====
        net_g.eval()
        avg_psnr = 0.0
        avg_ssim = 0.0
        for test_iter, batch in enumerate(data_loader_test, 1):
            rgb_input, target, idx = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2]
            prediction = net_g(rgb_input)
            out = torch.cat((rgb_input, prediction, target), 3)
            save_img(out[0].detach().cpu(), os.path.join(img_test, f'{idx[0]}.png'))
            psnr = torchPSNR(prediction, target).item()
            avg_psnr += psnr
            avg_ssim += ssim(prediction, target).item()

        avg_psnr /= len(data_loader_test)
        avg_ssim /= len(data_loader_test)
        print(f"===> Avg. PSNR: {avg_psnr:.4f} dB, Avg. SSIM: {avg_ssim:.4f}")

        # ===== CSV log per epoch =====
        row = {
            'run_tag': opt.run_tag,
            'epoch': epoch,
            'params_m': f"{params_m:.4f}",
            'latency_ms': f"{latency_ms:.2f}" if latency_ms is not None else "",
            'avg_psnr': f"{avg_psnr:.4f}",
            'avg_ssim': f"{avg_ssim:.4f}",
            'batch_size': opt.batch_size,
            'height': opt.hight,
            'width': opt.width,
            'seed': opt.seed
        }
        append_csv_row(opt.csv_out, row)
        print(f"↳ Logged to {opt.csv_out}: {row}")

        # ===== Checkpoints (go into per-run folders) =====
        if avg_psnr > best_psnr or avg_ssim > best_ssim:
            best_psnr = max(best_psnr, avg_psnr)
            best_ssim = max(best_ssim, avg_ssim)
            best_path = os.path.join(ckpt_best, f"netG_model_best_mixed_psnr_{best_psnr:.4f}_ssim_{best_ssim:.4f}.pth")
            torch.save(net_g, best_path)
            print(f"Best model saved at {best_path}")

        ckpt_path = os.path.join(ckpt_dir, f"netG_model_mixed_epoch_{epoch}_psnr_{avg_psnr:.4f}.pth")
        torch.save(net_g, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
