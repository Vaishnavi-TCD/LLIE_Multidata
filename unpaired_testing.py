# #!/usr/bin/env python3
# """
# unpaired_testing.py

# Run sliding-window inference on an **unpaired** image folder (e.g. DCIM) using a trained my_model.
# Produces side-by-side concatenated input | restored output at full resolution.
# """
# import os
# import argparse
# from glob import glob

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image

# # import your network class and save_img helper
# from model import my_model
# try:
#     from utils import save_img
# except ImportError:
#     from torchvision.utils import save_image as save_img


# def sw_inference(input_tensor, net, ph, pw, overlap):
#     """
#     Sliding-window inference with padding for arbitrary sizes:
#       input_tensor: [1,3,H,W] in [-1,1]
#       net: generator model
#       ph,pw: patch height & width
#       overlap: pixels overlap
#       returns: [1,3,H0,W0] restored output cropped to original
#     """
#     B, C, H0, W0 = input_tensor.shape
#     # 1) pad to minimum patch size
#     pad_h = max(ph - H0, 0)
#     pad_w = max(pw - W0, 0)
#     # 2) pad to even dimensions for pixel_unshuffle compatibility
#     pad_h += (H0 + pad_h) % 2
#     pad_w += (W0 + pad_w) % 2
#     # padding order: (left, right, top, bottom)
#     input_padded = F.pad(input_tensor,
#                          (0, pad_w, 0, pad_h), mode='reflect')
#     _, _, H, W = input_padded.shape

#     stride_h = ph - overlap
#     stride_w = pw - overlap

#     output = torch.zeros_like(input_padded)
#     weight = torch.zeros_like(input_padded)

#     ys = list(range(0, H, stride_h))
#     xs = list(range(0, W, stride_w))
#     if ys[-1] + ph > H: ys[-1] = H - ph
#     if xs[-1] + pw > W: xs[-1] = W - pw

#     with torch.no_grad():
#         for y in ys:
#             for x in xs:
#                 patch = input_padded[:, :, y:y+ph, x:x+pw]
#                 out_p = net(patch)
#                 output[:, :, y:y+ph, x:x+pw] += out_p
#                 weight[:, :, y:y+ph, x:x+pw] += 1.0

#     output = output / weight
#     # 3) crop back to original H0,W0
#     return output[:, :, :H0, :W0]


# class UnpairedTestDataset(Dataset):
#     """Loads all images in a single folder (jpg/JPG/png)."""
#     def __init__(self, root, transform):
#         exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
#         paths = []
#         for ext in exts:
#             paths.extend(glob(os.path.join(root, ext)))
#         self.paths = sorted(paths)
#         if not self.paths:
#             raise RuntimeError(f"No images found in {root}")
#         print(f"[INFO] Found {len(self.paths)} images in {root}")
#         self.transform = transform

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         img  = Image.open(path).convert('RGB')
#         return self.transform(img), os.path.basename(path)


# def run_inference(ckpt_path, test_root, out_root, device, opts):
#     os.makedirs(out_root, exist_ok=True)
#     print(f"\n[RUNNING] {os.path.basename(ckpt_path)} on {test_root}")

#     # load model
#     net = my_model().to(device)
#     ckpt_obj = torch.load(ckpt_path, map_location=device)
#     if isinstance(ckpt_obj, my_model):
#         net = ckpt_obj.to(device)
#     elif isinstance(ckpt_obj, dict):
#         sd = ckpt_obj.get('state_dict', ckpt_obj)
#         net.load_state_dict(sd)
#     else:
#         net.load_state_dict(ckpt_obj)
#     net.eval()

#     # transforms: to tensor + normalize, keep full resolution
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
#     ])

#     dataset = UnpairedTestDataset(test_root, transform)
#     loader  = DataLoader(dataset, batch_size=1, shuffle=False,
#                          num_workers=opts.num_workers)

#     with torch.no_grad():
#         for inp, fname in loader:
#             inp = inp.to(device)
#             # full-res sliding-window inference
#             pred = sw_inference(inp, net,
#                                 opts.height, opts.width,
#                                 opts.overlap)

#             # undo normalization
#             inp_vis  = (inp  * 0.5 + 0.5).clamp(0,1)
#             pred_vis = (pred * 0.5 + 0.5).clamp(0,1)
#             out = torch.cat([inp_vis, pred_vis], dim=3)

#             if isinstance(fname, (list, tuple)):
#                 fname = fname[0]
#             save_path = os.path.join(out_root, fname)
#             save_img(out[0].cpu(), save_path)

#     print(f"[DONE] Outputs saved to {out_root}")


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--cuda',       action='store_true')
#     p.add_argument('--height',     type=int, default=256,
#                    help='patch height')
#     p.add_argument('--width',      type=int, default=384,
#                    help='patch width')
#     p.add_argument('--overlap',    type=int, default=32,
#                    help='sliding-window overlap')
#     p.add_argument('--num_workers',type=int, default=0)
#     opt = p.parse_args()

#     device = torch.device('cuda' if opt.cuda and torch.cuda.is_available()
#                           else 'cpu')

#     # user-configured checkpoints, input folder, and output folder
#     configs = [
#     (
#         "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/real/netG_model_best_real_valPSNR_27.0532_SSIM_0.9773.pth",
#         "Z:/Documents/Low_light_image_restoration/Datasets/Unpaired_Dataset/LDR_TEST_IMAGES_DICM/LDR_TEST_IMAGES_DICM",
#         "./unpaired_outputs/dcim_lolv2_real"
#     ),
#         # add more tuples as needed
#     ]

#     for ckpt, test_root, out_dir in configs:
#         run_inference(ckpt, test_root, out_dir, device, opt)

# if __name__ == '__main__':
#     main()









#!/usr/bin/env python3
import os, glob, argparse
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import my_model
from torchvision.utils import save_image as save_img

class UnpairedTestDataset(Dataset):
    def __init__(self, root, transform):
        # pick up JPG/JPEG/PNG in that single folder
        exts = ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG")
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, e))
        self.paths = sorted(self.paths)
        if len(self.paths)==0:
            raise RuntimeError(f"No images found in {root}")
        print(f"[INFO] Found {len(self.paths)} images in {root}")
        self.transform = transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), os.path.basename(p)

# def sw_inference(x, net, ph, pw, overlap):
#     """
#     Sliding-window inference on 1×3×H×W tensor.
#     x should already be padded so H and W are even.
#     Returns full 1×3×H×W output.
#     """
#     B,C,H,W = x.shape
#     stride_h, stride_w = ph-overlap, pw-overlap

#     # accumulators
#     out = torch.zeros_like(x)
#     wgt = torch.zeros_like(x)

#     # tile positions
#     ys = list(range(0, H, stride_h))
#     xs = list(range(0, W, stride_w))
#     if ys[-1]+ph > H: ys[-1] = H-ph
#     if xs[-1]+pw > W: xs[-1] = W-pw

#     with torch.no_grad():
#         for y in ys:
#             for x0 in xs:
#                 patch = x[:,:, y:y+ph, x0:x0+pw]
#                 p_out = net(patch)
#                 out[:,:, y:y+ph, x0:x0+pw] += p_out
#                 wgt[:,:, y:y+ph, x0:x0+pw] += 1.0

#     return out.div_(wgt)
import math
import torch
import torch.nn.functional as F

def sw_inference(inp, net, ph, pw, overlap):
    """
    Sliding-window inference with:
      - reflect-padding so patch dims >= ph,pw and even
      - ceil-based tiling so there's no uncovered “black” regions
      - final crop back to original H0×W0
    """
    B, C, H0, W0 = inp.shape

    # --- 1) pad so H,W >= patch size and even ---
    pad_h_min  = max(ph - H0, 0)
    pad_w_min  = max(pw - W0, 0)
    pad_h_extra = (H0 + pad_h_min) % 2
    pad_w_extra = (W0 + pad_w_min) % 2
    pad_h = pad_h_min + pad_h_extra
    pad_w = pad_w_min + pad_w_extra

    if pad_h or pad_w:
        # F.pad takes (left, right, top, bottom)
        inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

    # new dims
    _, _, H, W = inp.shape
    stride_h = ph - overlap
    stride_w = pw - overlap

    # --- 2) figure out how many tiles to cover every pixel ---
    n_h = math.ceil((H - overlap) / stride_h)
    n_w = math.ceil((W - overlap) / stride_w)

    # prepare accumulators
    output = torch.zeros_like(inp)
    weight = torch.zeros_like(inp)

    # --- 3) run all patches, snapping last tile to the edge ---
    for i in range(n_h):
        for j in range(n_w):
            y0 = min(i * stride_h, H - ph)
            x0 = min(j * stride_w, W - pw)

            patch = inp[:, :, y0:y0 + ph, x0:x0 + pw]
            with torch.no_grad():
                p_out = net(patch)

            output[:, :, y0:y0 + ph, x0:x0 + pw] += p_out
            weight[:, :, y0:y0 + ph, x0:x0 + pw]  += 1.0

    # --- 4) normalize and crop back to H0,W0 ---
    output = output / weight
    output = output[:, :, :H0, :W0]
    return output



def run_inference(ckpt, test_root, out_dir, device, ph, pw, overlap):
    print(f"\n[RUNNING] {os.path.basename(ckpt)} on {test_root}")
    os.makedirs(out_dir, exist_ok=True)

    # load model
    net = my_model().to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    if isinstance(ckpt_obj, my_model):
        net = ckpt_obj.to(device)
    elif isinstance(ckpt_obj, dict):
        sd = ckpt_obj.get("state_dict", ckpt_obj)
        net.load_state_dict(sd)
    else:
        net.load_state_dict(ckpt_obj)
    net.eval()

    # data loader: no Resize, just ToTensor+Normalize
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    loader = DataLoader(
        UnpairedTestDataset(test_root, tf),
        batch_size=1, shuffle=False, num_workers=0
    )

    for inp, fname in loader:
        # 1) pad to even dimensions
        _,_,H0,W0 = inp.shape
        pad_h = (H0%2)
        pad_w = (W0%2)
        if pad_h or pad_w:
            inp = F.pad(inp, (0,pad_w, 0,pad_h), mode="reflect")

        inp = inp.to(device)    # now 1×3×H×W with H,W even
        # 2) sliding-window
        pred = sw_inference(inp, net, ph, pw, overlap)
        # 3) unpad back to original size
        pred = pred[:,:,:H0, :W0]
        inp  = inp [:,:,:H0, :W0]

        # 4) undo normalize and concat
        inp_vis  = (inp  *0.5 +0.5).clamp(0,1)
        pred_vis = (pred *0.5 +0.5).clamp(0,1)
        out = torch.cat([inp_vis, pred_vis], dim=3)

        save_image = os.path.join(out_dir, fname[0] if isinstance(fname,(list,tuple)) else fname)
        save_img(out[0].cpu(), save_image)

    print(f"[DONE] outputs saved to {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--height",  type=int, default=256)
    p.add_argument("--width",   type=int, default=384)
    p.add_argument("--overlap", type=int, default=32)
    opts = p.parse_args()

    device = torch.device("cuda" if opts.cuda and torch.cuda.is_available() else "cpu")

    # EDIT these paths for your setup:
    CONFIGS = [
    (
        "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/real/netG_model_best_real_valPSNR_27.0532_SSIM_0.9773.pth",
        "Z:/Documents/Low_light_image_restoration/Datasets/Unpaired_Dataset/LDR_TEST_IMAGES_DICM/LDR_TEST_IMAGES_DICM",
        "./unpaired_outputs/dcim_lolv2_real"
    ),
    ]

    for ck, test_root, out_dir in CONFIGS:
        run_inference(ck, test_root, out_dir, device,
                      ph=opts.height, pw=opts.width, overlap=opts.overlap)

if __name__ == "__main__":
    main()

