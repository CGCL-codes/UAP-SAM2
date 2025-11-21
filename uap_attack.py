import copy
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from imagecorruptions import corrupt
from torch.autograd import grad
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm, trange


from sam2_util import get_frame_index, collate_fn, choose_dataset, get_video_to_indices, load_model, \
    get_fused_prototype, infonce_loss
from attack_setting import SamForwarder, make_multi_prompts,seed_everything
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Your script description here")
    parser.add_argument('--limit_img', default=100, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--limit_frames', default=15, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--fea_num', default=30, type=int)
    parser.add_argument('--train_dataset', default='YOUTUBE')
    parser.add_argument('--test_dataset', default='YOUTUBE')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--checkpoints', default='sam2-t', help='model checkpoint')

    parser.add_argument('--seed', default=30, type=int, help='rand seed')
    parser.add_argument('--eps', default = 10 / 255, type=float)
    parser.add_argument('--alpha', default= 2 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)
    parser.add_argument('--weight_fea', default=0.000001, type=float)

    parser.add_argument('--loss_fea', action='store_true')
    parser.add_argument('--loss_diff', action='store_true')
    parser.add_argument('--loss_t', action='store_true')



    return parser
def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args

if __name__ == '__main__':
    parser = get_parser()
    args = get_args(parser)
    seed_everything(seed=args.seed)

    device = "cuda:1"

    sam_fwder, predictor = load_model(args, device=device)

    custom_dataset = choose_dataset(args)
    video_to_indices = get_video_to_indices(custom_dataset)
    data_loader = DataLoader(custom_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    denorm = lambda x: sam_fwder.denorm_image(x)
    weight_Y = -1

    loss_fn = F.mse_loss
    mse_loss = torch.nn.MSELoss()
    cosine_loss = F.cosine_similarity
    cosfn = torch.nn.CosineSimilarity(dim=-1)

    tensor_shape = (1, 3, 1024, 1024)
    shape_tensor = torch.empty(tensor_shape)

    feature_diff = 0
    loss_fea = 0
    loss_t = 0
    loss_ft = 0

    weight_loss_fea = 0
    weight_loss_diff = 0
    weight_loss_t = 0

    perturbation = torch.empty_like(shape_tensor).uniform_(-args.eps, args.eps).to(device)

    sample_step_count = 0
    sample_total_g = torch.zeros_like(perturbation, device=device)

    prev_adv_feature = None

    target_image_dir = './data/sav_test/JPEGImages_24fps'


    folders = [f for f in os.listdir(target_image_dir) if os.path.isdir(os.path.join(target_image_dir, f))]

    if len(folders) >= args.fea_num:
        selected_folders = random.sample(folders, args.fea_num)
    else:
        selected_folders = folders

    for step in range(args.P_num):
        for video_name, indices in video_to_indices.items():
            print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

            folder = random.choice(selected_folders)
            folder_path = os.path.join(target_image_dir, folder)
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = random.choice(image_files)
                image = Image.open(image_path).convert("RGB")
                image = image.resize((1024, 1024), Image.Resampling.BICUBIC)
                image = np.array(image)
                tgt = sam_fwder.transform_image(image).to(device)
                tgt = denorm(tgt)
                target_feature = sam_fwder.get_image_feature(tgt)
            else:
                print("No images found in the selected folder.")
                continue

            video_subset = Subset(custom_dataset, indices)
            video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

            pre_dict = None
            pre_dict_adv = None
            mask_pre = None
            mask_pre_adv = None
            start_frame_processed = False

            for images, P_list, img_ids, gt, point in tqdm(video_loader):
                img_ID, img, mask_gt, P_gt = img_ids[0], images[0], gt[0], P_list[0]
                video_name = img_ID.split('/')[0]
                frame_idx = get_frame_index(img_ID)

                X = sam_fwder.transform_image(img).to(device)
                benign_img = denorm(X)
                H, W, _ = img.shape
                Y = torch.ones([1, 1, H, W]).to(X.device, torch.float32) * weight_Y
                Y_bin = Y.bool()
                assert Y_bin.dtype in ['bool', bool, torch.bool]
                print(f"args.train_dataset: {args.train_dataset} ")

                transform1 = transforms.RandomRotation(degrees=15)
                transform2 = transforms.Lambda(lambda img: img + 0.03 * torch.rand_like(img))
                transform3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)

                aug_img1 = transform1(benign_img)
                aug_img2 = transform2(benign_img)
                aug_img3 = transform3(benign_img)

                img_list = [aug_img1, aug_img2, aug_img3]
                prototype_feature = get_fused_prototype(img_list, sam_fwder, device)
                prototype_feature = prototype_feature.to(device)

                if not start_frame_processed:
                    start_frame_idx = frame_idx
                    pre_dict = None
                    pre_dict_adv = None
                    mask_pre = None
                    mask_pre_adv = None
                    start_frame_processed = True
                    start_P = P_gt

                prompts = make_multi_prompts(args.point, (1024, 1024), args.prompts_num)
                P = sam_fwder.transform_prompts(*prompts)

                logits_clean = sam_fwder.forward(benign_img, *P)
                mask_clean = logits_clean > sam_fwder.mask_threshold

                mask_pre = mask_clean.clone().detach()
                output_dict = sam_fwder.get_current_out(frame_idx, benign_img, mask_pre)
                pre_dict = output_dict

                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True

                logits = sam_fwder.forward(adv_img, *P)
                mask = logits > sam_fwder.mask_threshold

                mask_pre_adv = mask.clone().detach()
                output_dict_adv = sam_fwder.get_current_out(frame_idx, adv_img, mask_pre_adv)
                pre_dict_adv = output_dict_adv

                adv_feature = sam_fwder.get_image_feature(adv_img)


                if args.loss_t:
                    attacked = mask == Y_bin
                    output = attacked * logits
                    output_f = ~attacked * (1 - logits)
                    loss_t = F.binary_cross_entropy_with_logits(output, Y)
                    loss_ft = -F.binary_cross_entropy_with_logits(output_f, Y)

                    weight_loss_t = 1

                if args.loss_diff:
                    if prev_adv_feature is not None:
                        feature_diff = -cosine_loss(prev_adv_feature, adv_feature).mean()
                        weight_loss_diff = 1
                    else:
                        feature_diff = 0

                if args.loss_fea:
                    loss_fea = infonce_loss(adv_feature, prototype_feature, target_feature)

                loss = weight_loss_t * loss_t + 0.01 * loss_ft  +  weight_loss_diff * feature_diff + args.weight_fea*loss_fea

                g = grad(loss, adv_img, loss)[0]

                beta = 0.95
                if step == 0:
                    ema_grad = g.detach()
                else:
                    ema_grad = beta * ema_grad + (1 - beta) * g.detach()
                sample_total_g += ema_grad
                sample_step_count += 1
                if sample_step_count > 0:
                    avg_gradient = sample_total_g / sample_step_count
                    perturbation = (perturbation - avg_gradient.sign() * args.alpha).clamp(-args.eps, args.eps).detach()
                prev_adv_feature = adv_feature.detach()

    uap_save_path = f"uap_file/{args.train_dataset}.pth"
    torch.save(perturbation.cpu(), uap_save_path)
    print(f"\n Global UAP saved to {uap_save_path}")

























