import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.profiler import record_function

from torchvision.transforms import functional as TF

import kornia

import time
import os
import slip
from tqdm import tqdm
from timit import timit
import numpy as np
from perlin_numpy import generate_fractal_noise_2d
from PIL import ImageFile, Image, PngImagePlugin

from typing import Tuple


class_table = {
    
}
try:
    from pixeldrawer import PixelDrawer
    from fast_pixeldrawer3 import FastPixelDrawer
    # update class_table if these import OK
    class_table.update({
        "pixel": PixelDrawer,
        "fast_pixel": FastPixelDrawer
    })
except ImportError as e:
    print("--> Not running with pydiffvg drawer support ", e)
    pass

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

class Prompt2(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.register_buffer('embed', embed)

    def forward(self, input):
        input_adj = input.unsqueeze(1)
        embed_adj = self.embed.unsqueeze(0)
        s = sum(sum(torch.cosine_similarity(input_adj,embed_adj)))
        return s


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        global aspect_ratio

        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cutn_zoom = int(0.6*cutn)
        self.cut_pow = cut_pow
        self.transforms = None


       
        augmentations = []
        augmentations.append(transforms.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.25,0.95),  ratio=(0.85,1.2),interpolation=transforms.InterpolationMode.NEAREST))
        augmentations.append(transforms.ColorJitter(hue=0.1, saturation=0.1))
        self.augs_zoom = nn.Sequential(*augmentations)

        augmentations = []
        if aspect_ratio == 1:
            n_s = 0.95
            n_t = (1-n_s)/2
            augmentations.append(transforms.RandomAffine(degrees=0, translate=(n_t, n_t), scale=(n_s, n_s),interpolation=transforms.InterpolationMode.NEAREST))
        elif aspect_ratio > 1:
            n_s = 1/aspect_ratio
            n_t = (1-n_s)/2
            augmentations.append(transforms.RandomAffine(degrees=0, translate=(0, n_t), scale=(0.9*n_s, n_s),interpolation=transforms.InterpolationMode.NEAREST))
        else:
            n_s = aspect_ratio
            n_t = (1-n_s)/2
            augmentations.append(transforms.RandomAffine(degrees=0, translate=(n_t, 0), scale=(0.9*n_s, n_s),interpolation=transforms.InterpolationMode.NEAREST))

        augmentations.append(transforms.CenterCrop(size=self.cut_size))
        augmentations.append(transforms.ColorJitter(hue=0.1, saturation=0.1))
        self.augs_wide = nn.Sequential(*augmentations)

        self.noise_fac = 0
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        global aspect_ratio, cur_iteration
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        mask_indexes = None


        with timit.time("cutmk"), record_function("cutmk"):
            for _ in range(self.cutn):
                # Pooling
                cutout = (self.av_pool(input) + self.max_pool(input))/2

                if mask_indexes is not None:
                    cutout[0][mask_indexes] = 0.0 # 0.5

                if aspect_ratio != 1:
                    if aspect_ratio > 1:
                        cutout = kornia.geometry.transform.rescale(cutout, (1, aspect_ratio))
                    else:
                        cutout = kornia.geometry.transform.rescale(cutout, (1/aspect_ratio, 1))

                cutouts.append(cutout)

        with timit.time("cuttf"), record_function("cuttf"):
            if self.transforms is not None:
                # print("Cached transforms available")
                batch1, batch2 = None,None
                with timit.time("cut-ctf-1"):
                    batch1 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[:self.cutn_zoom], dim=0), self.transforms[:self.cutn_zoom],
                        (self.cut_size, self.cut_size), padding_mode="fill")
                with timit.time("cut-ctf-2"):
                    batch2 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[self.cutn_zoom:], dim=0), self.transforms[self.cutn_zoom:],
                        (self.cut_size, self.cut_size), padding_mode="fill")
                batch = torch.cat([batch1, batch2])
                # if cur_iteration < 2:
                #     for j in range(4):
                #         TF.to_pil_image(batch[j].cpu()).save(f"cached_im_{cur_iteration:02d}_{j:02d}_{spot}.png")
                #         j_wide = j + self.cutn_zoom
                #         TF.to_pil_image(batch[j_wide].cpu()).save(f"cached_im_{cur_iteration:02d}_{j_wide:02d}_{spot}.png")
            else:
                if False:
                    batch1 = None
                    batch2 = None
                    with timit.time("cut-zoom"), record_function("cut1"):
                        batch1 = self.augs_zoom(torch.cat(cutouts[:self.cutn_zoom], dim=0))
                    with timit.time("cut-wide"), record_function("cut2"):
                        batch2 = self.augs_wide(torch.cat(cutouts[self.cutn_zoom:], dim=0))
                    # print(batch1.shape, batch2.shape)
                    batch = torch.cat([batch1, batch2])
                else:
                    if self.cutn == 1:
                        with timit.time("cut-wide"), record_function("cut"):
                            batch, transforms = self.augs_wide(cutouts[0])
                    else:
                        batch1, transforms1, batch2, transforms2 = None,None,None,None
                        with timit.time("cut-zoom"), record_function("cut1"):
                            batch1 = self.augs_zoom(torch.cat(cutouts[:self.cutn_zoom], dim=0))
                        with timit.time("cut-wide"), record_function("cut2"):
                            batch2 = self.augs_wide(torch.cat(cutouts[self.cutn_zoom:], dim=0))
                        batch = torch.cat([batch1, batch2])
                        #self.transforms = torch.cat([transforms1, transforms2])

                        ## batch, self.transforms = self.augs(torch.cat(cutouts, dim=0))
                        # if cur_iteration < 4:
                        #     for j in range(4):
                        #         TF.to_pil_image(batch[j].cpu()).save(f"live_im_{cur_iteration:02d}_{j:02d}_{spot}.png")
                        #         j_wide = j + self.cutn_zoom
                        #         TF.to_pil_image(batch[j_wide].cpu()).save(f"live_im_{cur_iteration:02d}_{j_wide:02d}_{spot}.png")

        bounds = [0,1]

        if self.noise_fac:
            with timit.time("cut-noise"):
                facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
                ran = torch.randn_like(batch)
                batch = batch + facs * ran
                bounds = [int(batch.min()),int(batch.max())]
        return batch, bounds

class MakeCutouts2(nn.Module):
    def __init__(self, modelDim):
        super().__init__()
        
        self.proc = transforms.Compose([transforms.RandomVerticalFlip(), transforms.RandomRotation(30), transforms.RandomPerspective(distortion_scale=0.2),transforms.RandomCrop(modelDim)])
        #self.proc = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)

    def forward(self, input):
        return self.proc(input), [0,1]

def contrast_noise(n):
    n = 0.9998 * n + 0.0001
    n1 = (n / (1-n))
    n2 = np.power(n1, -2)
    n3 = 1 / (1 + n2)
    return n3
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def random_noise_image(w,h):
    print("generating random noise image: {}x{}".format(w,h))
    # scale up roughly as power of 2
    if (w>1024 or h>1024):
        side, octp = 2048, 6
    elif (w>512 or h>512):
        side, octp = 1024, 5
    elif (w>256 or h>256):
        side, octp = 512, 4
    else:
        side, octp = 256, 3

    nr = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    ng = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    nb = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    stack = np.dstack((contrast_noise(nr),contrast_noise(ng),contrast_noise(nb)))
    substack = stack[:h, :w, :]
    im = Image.fromarray((255.999 * substack).astype('uint8'))
    return im

drawer = None
device = None
size = None
pixel_size = None
perceptors = None
prompt_table = {}
opts = None

run_id = None

aspect_ratio = None

clip_models = ["ViT-B/16","ViT-B/32","RN50"]
num_loss_drop = 0
learning_rate = 0.3/2
num_batches = 1
num_cuts = 10
cutouts_table = {}
cutouts_size_table = {}
save_every = 10
loss_every = 1

cur_iteration = 0

def do_init(drawer_str: str, size_: Tuple[int, int], pixel_size_:Tuple[int,int], prompt:str):
    global drawer,device,size,perceptors,prompt_table,opts, clip_models, pixel_size
    global num_cuts, aspect_ratio, run_id, learning_rate

    run_id = int(time.time())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    size = size_
    pixel_size = pixel_size_
    aspect_ratio = size[0] / size[1]

    class Settings(object):
        def __init__(self,device,size,pixel_size):
            self.device = device
            self.size = size
            self.pixel_size = pixel_size

    drawer = class_table[drawer_str](Settings(device,size,pixel_size_), learning_rate)
    drawer.load_model(None,device)

    img = random_noise_image(size[0], size[1])
    starting_tensor = TF.to_tensor(img)
    init_tensor = starting_tensor.to(device).unsqueeze(0)
    drawer.init_from_tensor(init_tensor * 2 - 1)

    perceptors = {}
    for clip_model in clip_models:
        perceptor = slip.get_clip_perceptor(clip_model, device)
        perceptors[clip_model] = perceptor

        prompt_table[clip_model] = []

        cut_size = perceptor.input_resolution
        cutouts_size_table[clip_model] = cut_size
        if not cut_size in cutouts_table:
            cutouts_table[cut_size] = MakeCutouts(cut_size, num_cuts)
            #cutouts_table[cut_size] = MakeCutouts2(cut_size)
    
    for clip_model in clip_models:
        perceptor = perceptors[clip_model]

        embed = perceptor.encode_text(prompt).float()
        prompt_table[clip_model].append(Prompt(embed))
    
    opts = rebuild_optimisers()

def do_run(iterations: int):
    global drawer, cur_iteration, save_every

    with tqdm() as pbar:
        while True:
            with timit.time("train"):
                keep_going = train()
                cur_iteration += 1

                if cur_iteration%save_every == 0 or cur_iteration>=iterations or not keep_going:
                    save_cur_img()
                    

                if cur_iteration >= iterations or not keep_going:
                    break

                pbar.update()

def rebuild_optimisers():
    global drawer,num_loss_drop,learning_rate

    drop_divisor = 10 ** num_loss_drop
    opt = drawer.get_opts(drop_divisor)

    if opt == None:
        dropped_learning_rate = learning_rate/drop_divisor
        to_optimize = [ drawer.get_z() ]
        opt = [optim.Adam(to_optimize, lr=dropped_learning_rate)]

    return opt

def gen_loss():
    global drawer, cur_iteration

    out = drawer.synth(cur_iteration)

    result = []
    result_labled = {}

    cur_cutouts = {}
    cur_cutouts_orig = {}
    with timit.time("cut"), record_function("cut"):
        for cutoutSize in cutouts_table:
            res, bounds = cutouts_table[cutoutSize](out)
            cur_cutouts[cutoutSize] = [res,bounds]
            cur_cutouts_orig[cutoutSize] = res

    for clip_model in clip_models:
        perceptor = perceptors[clip_model]
        cutoutSize = cutouts_size_table[clip_model]

        iii = None
        with timit.time("enc"), record_function("enc"):
            if True:
                [cutout, bounds] = cur_cutouts[cutoutSize]
                iii = perceptor.encode_image(cutout,input_range=bounds).float()
                #iii = perceptor.encode_image(out).float()
                # if cur_iteration%save_every == 0:
                #     save_image(cutout, clip_model.replace("/","_"))
            else:
                if cutoutSize not in nocut_proc:
                    nocut_proc[cutoutSize] = transforms.Compose([transforms.Resize((cutoutSize,cutoutSize))])
                iii = perceptor.encode_image(nocut_proc[cutoutSize](out),input_range=(0,1)).float()

        with timit.time("prompt"), record_function("prompt"):
            i = 0
            for prompt in prompt_table[clip_model]:
                res = prompt(iii)
                
                result.append(res)
                result_labled["m: {} pmpt{}".format(clip_model,i)] = float(res)
                i+=1

            out_ = drawer.synth(cur_iteration, "bilinear")
            # if cur_iteration%save_every == 0:
            #     save_image(out_,"_out_")
            for prompt in prompt_table[clip_model]:
                res = prompt(perceptor.encode_image(out_,input_range=[0,1]).float())
                
                result.append(res)
                result_labled["m: {} pmpt{}".format(clip_model,i)] = float(res)
                i+=1

    for cutoutSize in cutouts_table:
        # clear the transform "cache"
        cutouts_table[cutoutSize].transforms = None

    return result, result_labled


def train():
    global drawer, opts, cur_iteration


    for opt in opts:
        opt.zero_grad()

    for i in range(num_batches):
        with timit.time("ascend_txt"), record_function("ascend_txt"):
            lossAll, lossAllLabels = gen_loss()

        with timit.time("backward"), record_function("backward"):
            loss = sum(lossAll)
            loss.backward()

    if cur_iteration%loss_every == 0:
        tqdm.write("losses @{}: {} [{}]".format(cur_iteration, float(loss), lossAllLabels))

    with timit.time("step"), record_function("step"):
        for opt in opts:
            opt.step()

    drawer.clip_z()

    return True


def save_cur_img():
    global drawer, cur_iteration, run_id

    img = drawer.synth(cur_iteration)
    save_image(img)
    

def save_image(img, additional_name=""):
    img = TF.to_pil_image(img[0].cpu())

    dir_ = "out2/{}".format(run_id)
    path = dir_+"/{}.png".format(str(cur_iteration)+additional_name)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    img.save(path)

def main():
    timit.init()
    do_init("fast_pixel", (480,270), (80,45), "A beautiful sunset on a beach looking towards the ocean #pixelart #8bit")
    do_run(1200)
    timit.end()
    timit.print()


if __name__ == "__main__":
    main()