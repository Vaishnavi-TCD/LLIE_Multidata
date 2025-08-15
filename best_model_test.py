import os
import argparse
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import my_model       # your network definition
from utils import save_img       # your existing image‐saving helper

class PairedTestDataset(Dataset):
    def __init__(self, root, transform):
        low_dir    = os.path.join(root, "Low")
        normal_dir = os.path.join(root, "Normal")
        self.low_paths  = sorted(glob(os.path.join(low_dir, "*")))
        self.norm_paths = sorted(glob(os.path.join(normal_dir, "*")))
        assert len(self.low_paths) == len(self.norm_paths), \
            f"Mismatched Low vs Normal: {len(self.low_paths)} vs {len(self.norm_paths)}"
        self.transform = transform

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        lr = Image.open(self.low_paths[idx]).convert("RGB")
        hr = Image.open(self.norm_paths[idx]).convert("RGB")
        return self.transform(lr), self.transform(hr), os.path.basename(self.low_paths[idx])

def make_loader(root, batch_size, num_workers, height, width):
    # — exactly the same transforms you used in training —
    tf = transforms.Compose([
        transforms.Resize((height, width)),   # h×w
        transforms.ToTensor(),               # → [0,1]
        transforms.Normalize(                # → [-1,1]
            mean=(0.5, 0.5, 0.5),
            std =(0.5, 0.5, 0.5)
        ),
    ])
    ds = PairedTestDataset(root, tf)
    print(f"[INFO] Loading test set {root} → {len(ds)} images")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def run_checkpoint(ckpt_path, test_root, out_root, device, opts):
    os.makedirs(out_root, exist_ok=True)
    print(f"\n[RUNNING] {os.path.basename(ckpt_path)} on {test_root}")

    # 1) instantiate & load network
    net = my_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
        net.load_state_dict(sd)
    else:
        net = ckpt.to(device)
    net.eval()

    # 2) build DataLoader
    loader = make_loader(test_root,
                         batch_size=opts.batch_size,
                         num_workers=opts.num_workers,
                         height=opts.height,
                         width=opts.width)

    # 3) inference + undo‐normalize + save exactly as in training
    with torch.no_grad():
        for lr, hr, fname in loader:
            # to device
            lr = lr.to(device)    # in [-1,1]
            hr = hr.to(device)

            pred = net(lr)        # out in [-1,1]

            # — undo normalization (same as training) —
            # lr   = (lr   * 0.5 + 0.5).clamp(0,1)
            # pred = (pred * 0.5 + 0.5).clamp(0,1)
            # hr   = (hr   * 0.5 + 0.5).clamp(0,1)

            # concat input | output | GT horizontally
            triplet = torch.cat([lr, pred, hr], dim=3)  # [B,3,H,3W]

            # fname may be a list/tuple if batch_size>1
            if isinstance(fname, (list, tuple)):
                fname = fname[0]

            save_path = os.path.join(out_root, fname)
            save_img(triplet[0].cpu(), save_path)

    print(f"[DONE] outputs saved to {out_root}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda",       action="store_true", help="use GPU")
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--num_workers",type=int,   default=0)
    p.add_argument("--height",     type=int,   default=256, help="image height")
    p.add_argument("--width",      type=int,   default=384, help="image width")
    opt = p.parse_args()

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

   
# — YOUR collection of (checkpoint, test‐folder, output‐folder) —
    configs = [
        # (checkpoint,         test_folder,                                              output_folder)
        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/netG_model_best_mixed_psnr_22.6491_ssim_0.9951.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOLdataset/test",
        "./best_test_outputs/best_mixed_lolv1"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/netG_model_best_mixed_psnr_22.6491_ssim_0.9951.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test",
        "./best_test_outputs/best_mixed_lolv2_real"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/netG_model_best_mixed_psnr_22.6491_ssim_0.9951.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test",
        "./best_test_outputs/best_mixed_lolv2_syn"),



        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/v1/netG_model_best_v1_psnr_22.8075_ssim_0.9946.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOLdataset/test",
        "./best_test_outputs/best_lolv1_lolv1"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/v1/netG_model_best_v1_psnr_22.8075_ssim_0.9946.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test",
        "./best_test_outputs/best_lolv1_lolv2_real"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/v1/netG_model_best_v1_psnr_22.8075_ssim_0.9946.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test",
        "./best_test_outputs/best_lolv1_lolv2_syn"),


        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/real/netG_model_best_real_valPSNR_27.0532_SSIM_0.9773.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOLdataset/test",
        "./best_test_outputs/best_lolv2_real_lolv1"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/real/netG_model_best_real_valPSNR_27.0532_SSIM_0.9773.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test",
        "./best_test_outputs/best_lolv2_real"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/real/netG_model_best_real_valPSNR_27.0532_SSIM_0.9773.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test",
        "./best_test_outputs/best_lolv2_real_lolv2_syn"),


        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/syn/netG_model_best_syn_psnr_22.4103_ssim_0.9955.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOLdataset/test",
        "./best_test_outputs/best_lolv2_syn_lolv1"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/syn/netG_model_best_syn_psnr_22.4103_ssim_0.9955.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test",
        "./best_test_outputs/best_lolv2_syn_lolv2_real"),

        ("Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/syn/netG_model_best_syn_psnr_22.4103_ssim_0.9955.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Synthetic/Test",
        "./best_test_outputs/best_lolv2_syn_lolv2_syn"),
    ]


    for ckpt, test_root, out_dir in configs:
        run_checkpoint(ckpt, test_root, out_dir, device, opt)

if __name__ == "__main__":
    main()





























