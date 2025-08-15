# train_be.py
from __future__ import print_function
import argparse, os, time, csv, multiprocessing
import torch, torch.nn as nn, torch.optim as optim, torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import MS_SSIM, ssim
import kornia

from utils import save_img, VGGPerceptualLoss, torchPSNR
from network1 import GANLoss, get_scheduler, update_learning_rate

# IMPORTANT: B+E model
from model_be import my_model


def count_params_m(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_latency_ms(model, device, h, w, warmup=10, iters=100):
    model.eval(); x = torch.randn(1,3,h,w, device=device)
    with torch.no_grad():
        for _ in range(warmup): _ = model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters): _ = model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
    return (time.time()-t0)*1000.0/iters

def append_csv_row(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dirs, reference_dirs, transform=None, reference_transform=None):
        self.transform, self.reference_transform = transform, reference_transform
        val_ext = ('.jpg','.jpeg','.png','.bmp','.tiff')
        self.input_images, self.reference_images = [], []
        for inp, ref in zip(input_dirs, reference_dirs):
            if not (os.path.isdir(inp) and os.path.isdir(ref)): continue
            input_files = sorted([f for f in os.listdir(inp) if f.lower().endswith(val_ext)])
            reference_files = sorted([f for f in os.listdir(ref) if f.lower().endswith(val_ext)])
            for file in input_files:
                if file in reference_files:
                    self.input_images.append(os.path.join(inp, file))
                    self.reference_images.append(os.path.join(ref, file))
    def __len__(self): return len(self.input_images)
    def __getitem__(self, idx):
        inp = Image.open(self.input_images[idx]).convert("RGB")
        ref = Image.open(self.reference_images[idx]).convert("RGB")
        x = self.transform(inp) if self.transform else inp
        y = (self.reference_transform or self.transform)(ref)
        return x, y, idx

def get_dataset(root_dirs, h, w):
    t = transforms.Compose([transforms.Resize((h, w)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    return CustomImageDataset(root_dirs['input'], root_dirs['reference'], t)

parser = argparse.ArgumentParser(description='B+E Ablation (expansion-only) w/ CSV + per-run outputs')
parser.add_argument('--dataset', default='./LOL_data/')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--hight', type=int, default=256)
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--finetune', default=False)
parser.add_argument('--epoch_count', type=int, default=0)
parser.add_argument('--niter', type=int, default=60)
parser.add_argument('--niter_decay', type=int, default=40)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--lr_policy', type=str, default='lambda')
parser.add_argument('--lr_decay_iters', type=int, default=500)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--threads', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--edge_loss', default=False)
parser.add_argument('--edge_loss_type', default='sobel')
parser.add_argument('--run_tag', type=str, default='B+E_rank')
parser.add_argument('--csv_out', type=str, default='ablation_results.csv')

def main():
    opt = parser.parse_args(); print(opt)
    # per-run outputs
    outdir = os.path.join('outputs', opt.run_tag)
    img_train = os.path.join(outdir, 'images_train')
    img_test  = os.path.join(outdir, 'images_test')
    ckpt_best = os.path.join(outdir, 'best_checkpoint')
    ckpt_dir  = os.path.join(outdir, 'checkpoint', opt.dataset)
    for d in [img_train, img_test, ckpt_best, ckpt_dir]: os.makedirs(d, exist_ok=True)

    if opt.cuda and not torch.cuda.is_available(): print("→ No GPU; using CPU"); opt.cuda=False
    cudnn.benchmark=True; torch.manual_seed(opt.seed); 
    if opt.cuda: torch.cuda.manual_seed(opt.seed)
    device = torch.device('cuda:0' if opt.cuda else 'cpu')

    net_g = my_model().to(device)
    params_m = count_params_m(net_g)
    try: latency_ms = measure_latency_ms(net_g, device, opt.hight, opt.width)
    except Exception as e: print('[warn] latency failed:', e); latency_ms=None

    # data
    root_train = {'input': ['./lol_dataset/our485/low',
                            "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Low",
                            "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Low"],
                  'reference': ['./lol_dataset/our485/high',
                                "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Train/Normal",
                                "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Train/Normal"]}
    root_test  = {'input': ['./lol_dataset/eval15/low',
                            "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Low",
                            "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Low"],
                  'reference': ['./lol_dataset/eval15/high',
                                "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test/Normal",
                                "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test/Normal"]}
    dl_train = DataLoader(get_dataset(root_train, opt.hight, opt.width), batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.threads, pin_memory=opt.cuda)
    dl_test  = DataLoader(get_dataset(root_test,  opt.hight, opt.width), batch_size=1, shuffle=True,
                          num_workers=opt.threads, pin_memory=opt.cuda)

    # losses/optim (kept same as your trainer)
    class Gradient_Loss(nn.Module):
        def __init__(self):
            super().__init__()
            k=[[[0,1,0],[1,-4,1],[0,1,0]],[[0,1,0],[1,-4,1],[0,1,0]],[[0,1,0],[1,-4,1],[0,1,0]]]
            w=torch.FloatTensor(k).unsqueeze(0).permute(1,0,2,3); self.weight_g=nn.Parameter(w, requires_grad=False)
        def forward(self,x,xx):
            return nn.L1Loss()(F.conv2d(x,self.weight_g,groups=3), F.conv2d(xx,self.weight_g,groups=3))
    GradLoss = Gradient_Loss().to(device)
    L_per = VGGPerceptualLoss().to(device)
    MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
    Charb = nn.SmoothL1Loss().to(device)
    optim_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    sch_g   = get_scheduler(optim_g, opt)

    weights = {'Charbonnier':3.0,'Perceptual':2.0,'MS_SSIM':1.0,'Gradient':2.5,'Edge':0.5}

    print('===> Training')
    best_psnr=0; best_ssim=0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        net_g.train()
        for it,(rgb,tar,idx) in enumerate(dl_train,1):
            rgb,tar = rgb.to(device,non_blocking=True), tar.to(device,non_blocking=True)
            fake = net_g(rgb)
            optim_g.zero_grad()
            loss = (weights['Charbonnier']*Charb(fake,tar)
                    + weights['Perceptual']*L_per(fake,tar)
                    + weights['MS_SSIM']*(1-MS_SSIM_loss(fake,tar))
                    + weights['Gradient']*GradLoss(fake,tar))
            loss.backward(); optim_g.step()
            if it%100==0:
                out = torch.cat((rgb, fake, tar),3)
                save_img(out[0].detach().cpu(), os.path.join(img_train, f'{it}.png'))
                print(f"===> Epoch[{epoch}]({it}/{len(dl_train)}): Loss_G: {loss.item():.4f}")
        update_learning_rate(sch_g, optim_g)

        # eval
        net_g.eval(); avg_psnr=0.0; avg_ssim=0.0
        for test_it,(rgb,tar,idx) in enumerate(dl_test,1):
            rgb,tar = rgb.to(device,non_blocking=True), tar.to(device,non_blocking=True)
            pred = net_g(rgb)
            out = torch.cat((rgb, pred, tar),3)
            save_img(out[0].detach().cpu(), os.path.join(img_test, f'{idx[0]}.png'))
            avg_psnr += torchPSNR(pred, tar).item()
            avg_ssim += ssim(pred, tar).item()
        avg_psnr/=len(dl_test); avg_ssim/=len(dl_test)
        print(f"===> Avg. PSNR: {avg_psnr:.4f} dB, Avg. SSIM: {avg_ssim:.4f}")

        append_csv_row(opt.csv_out, {'run_tag':opt.run_tag,'epoch':epoch,'params_m':f"{params_m:.4f}",
                                     'latency_ms':f"{latency_ms:.2f}" if latency_ms is not None else "",
                                     'avg_psnr':f"{avg_psnr:.4f}",'avg_ssim':f"{avg_ssim:.4f}",
                                     'batch_size':opt.batch_size,'height':opt.hight,'width':opt.width,'seed':opt.seed})

        # checkpoints
        if avg_psnr>best_psnr or avg_ssim>best_ssim:
            best_psnr=max(best_psnr,avg_psnr); best_ssim=max(best_ssim,avg_ssim)
            best_path = os.path.join(ckpt_best, f"netG_best_psnr_{best_psnr:.4f}_ssim_{best_ssim:.4f}.pth")
            torch.save(net_g, best_path); print("Saved", best_path)
        ckpt_path = os.path.join(ckpt_dir, f"netG_epoch_{epoch}_psnr_{avg_psnr:.4f}.pth")
        torch.save(net_g, ckpt_path); print("Saved", ckpt_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
