import argparse
import os
import cv2
from math import log10
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import get_test_set
from model import my_model 

from utils import is_image_file, load_img, save_img
import time
torch.backends.cudnn.benchmark = True
from thop import profile



# Testing settings
parser = argparse.ArgumentParser(description='Spectroformer-implementation')
parser.add_argument('--dataset', default="Z:/Documents/Low_light_image_restoration/Datasets/LOL_V2/LOL-v2/Real_captured/Test", required=False, help='facades')
parser.add_argument('--save_path', default='./testing_code/Result/', required=False, help='facades')
parser.add_argument('--checkpoints_path', default='./checkpoints/blur/', required=False, help='facades')
parser.add_argument('--epoch_count', type=int, default=189,help='the starting epoch count')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', action='store_false', help='use cuda')
parser.add_argument('--show_flops_params',type=bool, default=True, help='Show number of flops and parameter of model')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
criterionMSE = nn.MSELoss().to(device)

os.makedirs(opt.save_path, exist_ok=True)
# G_path = "D:/Raqib/checkpoint/uw_data/netG_model_epoch_{}.pth".format(opt.epoch_count)
# G_path = "Z:/Documents/Low_light_image_restoration/LowLightMultiData/LowLIght/best_checkpoint/netG_model_best_mixed_psnr_22.6491_ssim_0.9951.pth"
# my_net = torch.load(G_path).to(device)


# my addition

G_path = os.path.join(opt.checkpoints_path,f"netG_model_epoch_{opt.epoch_count}.pth")
# 1) create the network and load weights
my_net = my_model(
    num_blocks=[1,1,3,3],
    num_heads =[1,1,2,4],
    channels  =[12,24,48,96],
    expansion_factor=2.0
).to(device)
state = torch.load(G_path, map_location=device)
my_net.load_state_dict(state)
my_net.eval()                             

if opt.show_flops_params:
    input_ = torch.randn (1, 3, 256, 256).cuda()
    flops, params = profile (my_net, inputs = (input_,))
    print("FLOPS of the network ARE::::::::::::::",flops)
    print("Parametrs of the network ARE::::::::::::::",params)    

# image_dir = "{}/".format(opt.dataset)
# image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

# transform_list = [transforms.ToTensor(),
#                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# transform = transforms.Compose(transform_list)

#myaddtion
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


test_set = get_test_set(opt.dataset)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

start = time.time()
avg_psnr=0
a = 0
times = []
# for iteration_test, batch in enumerate(testing_data_loader,1):
#     input1, filename = batch[0].to(device), batch[1]     
#     final_l = my_net(input1)[0]
#     print(final_l.shape)
#     # final_l=torch.cat([input1[0],final_l],dim=-1)   
#     final_l = final_l.detach().squeeze(0).cpu()
#     # print(filename[0])
#     save_img(final_l, opt.save_path+filename[0])

#myaddtion
with torch.no_grad():                      # <-- no grad for speed & memory
    for iteration_test, batch in enumerate(testing_data_loader, 1):
        input1, filename = batch[0].to(device), batch[1]
        # 2) ensure input is normalized the same way
        input1 = transform(input1)
        out = my_net(input1)[0]
        out = out.clamp(0.0, 1.0)          # clamp into [0,1]
        final_l = out.cpu().detach().squeeze(0)
        save_img(final_l, os.path.join(opt.save_path, filename[0]))

#         mse = criterionMSE(out, tar_clamped)
#         psnr = 10 * torch.log10(1.0 / mse)
#         avg_psnr += psnr.item()


# print("Avg. PSNR:", avg_psnr / len(testing_data_loader))
       










