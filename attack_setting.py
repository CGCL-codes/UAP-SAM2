import os
import random
from argparse import ArgumentParser, Namespace
from collections import deque
from typing import Tuple, Callable, Union, List
import numpy as np
import torch
import math
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import Module
from torch.autograd import Function
import pywt
from torchvision.transforms import Resize, Normalize
from numpy import ndarray
from numpy.typing import NDArray
from sam2.utils.misc import fill_holes_in_mask_scores
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

npimg_u8  = NDArray[np.uint8]
Size      = Tuple[int, int]
Point     = Tuple[int, int]
Prompts   = Tuple[ndarray, ndarray, None, None]
try:
    from moviepy.editor import ImageSequenceClip
    HAS_MOVIEPY = True
except ImportError:
    print('>> [warn] missing lib "moviepy", will not generate adv pred step by step')
    HAS_MOVIEPY = False
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam2_utils import select_closest_cond_frames, get_1d_sine_pe

CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']
LIM = ['', 'edge', 'smap', 'cam', 'tgt']
SAM_MASK_THRESH = 0.0

class SamForwarder(nn.Module):
    def __init__(self, sam: SAM2Base):
        super().__init__()
        self.model = sam
        self.device = 'cuda:1'
        self.mask_threshold = 0.0
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=self.mask_threshold,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.NO_OBJ_SCORE = -1024.0

        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.add_all_frames_to_correct_as_cond = False
        self.max_cond_frames_in_attn = -1
        self.num_maskmem = 7
        self.fill_hole_area = 0
        self.num_frames = 82
        assert SAM_MASK_THRESH == self.mask_threshold, f'sam.mask_threshold ({sam.mask_threshold}) != SAM_MASK_THRESH ({SAM_MASK_THRESH})'

    def norm_image(self, x: Tensor) -> Tensor:
        pixel_mean = torch.tensor(self.pixel_mean, device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=self.device).view(1, 3, 1, 1)
        return (x - pixel_mean) / pixel_std

    def denorm_image(self, x: Tensor) -> Tensor:
        pixel_mean = torch.tensor(self.pixel_mean, device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=self.device).view(1, 3, 1, 1)
        return x * pixel_std + pixel_mean

    def resize_image(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.canvas_size - w, 0, self.canvas_size - h))

    def unresize_image(self, x: Tensor) -> Tensor:
        INTERP_MODE = 'bilinear'
        align_corners = None if INTERP_MODE == 'nearest' else False
        x = F.interpolate(x, (self.canvas_size, self.canvas_size), mode=INTERP_MODE, align_corners=align_corners)
        x = x[..., :self.input_size[0], :self.input_size[1]]
        return F.interpolate(x, self.original_size, mode=INTERP_MODE, align_corners=align_corners)

    def transform_image(self, im: npimg_u8, is_edge: bool = False) -> Tensor:

        if isinstance(im, np.ndarray):
            self._orig_hw = [im.shape[:2]]
        x = self._transforms(im)
        X = x[None, ...].to(self.device)
        assert (len(X.shape) == 4 and X.shape[1] == 3), f"input_image must be of size 1x3xHxW, got {X.shape}"
        return X

    def transform_prompts(self, point_coords: np.ndarray = None, point_labels: np.ndarray = None,
                          box: np.ndarray = None,
                          mask_input: np.ndarray = None, normalize_coords=True) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        img_idx = -1
        unnorm_coords, labels_torch, unnorm_box, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            unnorm_coords = self._transforms.transform_coords(point_coords, normalize=normalize_coords,
                                                              orig_hw=self._orig_hw[img_idx])
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            unnorm_coords, labels_torch = unnorm_coords[None, ...], labels_torch[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            unnorm_box = unnorm_box.reshape(1, 1, 4)
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        return unnorm_coords, labels_torch, unnorm_box, mask_input_torch

    def forward(self, image: Tensor, point_coords: Tensor = None, point_labels: Tensor = None, boxes: Tensor = None,
                mask_input: Tensor = None, multi_mask: bool = False) -> Tensor:

        img_idx: int = -1
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True

        points = (point_coords, point_labels) if point_coords is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(points, boxes, mask_input)
        batched_mode = (points is not None and points[0].shape[0] > 1)
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, _, sam_token_out, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multi_mask,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        return masks

    def predict(self, frame_idx,images, previous_mask):
        backbone_out = self.model.forward_image(images)
        high_res_features = None
        if self.model.use_high_res_features_in_sam:
            high_res_features = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]
        _, _, _, _, _, obj_ptr, object_score_logits = self.model._forward_sam_heads(
            backbone_features=backbone_out["backbone_fpn"][-1],
            mask_inputs=previous_mask,
            high_res_features=high_res_features,
        )
        return obj_ptr,object_score_logits,high_res_features

    def get_image_feature(self, image: Tensor) -> Tensor:
        img_idx: int = -1
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self.features = self._features["image_embed"][img_idx].unsqueeze(0)
        return self.features

    def _run_single_frame_inference(
            self,
            frame_idx,
            image,
            output_dict,
            is_init_cond_frame,
            point_inputs,
            mask_inputs,
            prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        reverse = False
        run_mem_encoder = True
        backbone_out = self.model.forward_image(image)
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(
            backbone_out)
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        storage_device = self.device
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        pred_masks = self._transforms.postprocess_masks(pred_masks, self._orig_hw[-1])
        return pred_masks

    def get_current_out(self, frame_idx, image, pred_masks, output_dict=None, run_mem_encoder=True):
        obj_ptr, object_score_logits, high_res_features = self.predict(frame_idx, image, pred_masks)
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        backbone_out = self.model.forward_image(image)
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(
            backbone_out)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats,
            feat_sizes,
            pred_masks,
            object_score_logits,
            is_mask_from_pts=False,
        )
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        cond_storage_key = "cond_frame_outputs"
        output_dict[cond_storage_key][frame_idx] = compact_current_out
        non_cond_storage_key = "non_cond_frame_outputs"
        if frame_idx > 0 and frame_idx - 1 in output_dict[cond_storage_key]:
            output_dict[non_cond_storage_key][frame_idx - 1] = output_dict[cond_storage_key][frame_idx - 1]
        return output_dict

    def _prepare_memory_conditioned_features(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse=False,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        num_frames = self.num_frames
        B = current_vision_feats[-1].size(1)
        C = self.model.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.model.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            stride = 1 if self.model.training else self.model.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    if not track_in_reverse:
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        prev_frame_idx = frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = (
                        maskmem_enc + self.model.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.model.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.model.max_obj_ptrs_in_encoder)
                if not self.model.training and self.model.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.model.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.model.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.model.proj_tpos_enc_in_obj_ptrs else self.model.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.model.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.model.mem_dim)
                    if self.model.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.model.mem_dim, self.model.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.model.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.model.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
            to_cat_memory = [self.model.no_mem_embed.expand(1, B, self.model.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.model.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem, to_cat_memory
FwdPack = Tuple[SamForwarder, Prompts, Callable]

def seed_everything(seed: int):
    print('>> global seed:', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def _parse_point_gt(coord:str, size:Size) -> Point:
  if coord:
    point = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(point):
      if e < 1.0: point[i] = e * size[i]
      point[i] = int(point[i])
  else:
    point = [random.randrange(sz) for sz in size]
  return point
def make_prompts(point: Union[str, np.ndarray], img_size: tuple) -> Prompts:
    if isinstance(point, str) or point is None:
        point = _parse_point_gt(point, img_size)
        coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
    else:
        coords = point
    labels = np.asarray([1], dtype=np.int32)
    return (coords, labels, None, None)
def _parse_point(coord: Union[str, None], size: Size, prompt_num: int = 1) -> List[Point]:
    grid_size = math.ceil(math.sqrt(prompt_num))
    width, height = size
    x_step = width // grid_size
    y_step = height // grid_size

    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = np.random.randint(i * x_step, (i + 1) * x_step)
            y = np.random.randint(j * y_step, (j + 1) * y_step)
            grid_points.append((x, y))
    return grid_points

def make_multi_prompts(points: Union[str, np.ndarray], img_size: Size, prompt_num: int = 1) -> Prompts:
    if isinstance(points, str) or points is None:
        grid_points = _parse_point(points, img_size, prompt_num)
        coords = np.array(grid_points, dtype=np.int32)
        labels = np.ones((coords.shape[0],), dtype=np.int32)
    else:
        coords = np.asarray(points, dtype=np.int32)
        assert coords.ndim == 2 and coords.shape[1] == 2, "Points must have shape (N, 2)"

        if coords.shape[0] < prompt_num:
            grid_points = _parse_point(None, img_size, prompt_num)
            grid_points = np.array(grid_points, dtype=np.int32)
            coords = np.vstack([coords, grid_points[:prompt_num - coords.shape[0]]])
        elif coords.shape[0] > prompt_num:
            coords = coords[:prompt_num, :]

        labels = np.ones((coords.shape[0],), dtype=np.int32)

    return (coords, labels, None, None)

def _parse_box(coord: Union[str, None], size: Size, prompt_num: int = 1) -> List[Tuple[Point, Point]]:
    grid_size = math.ceil(math.sqrt(prompt_num))
    width, height = size
    x_step = width // grid_size
    y_step = height // grid_size

    grid_boxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            gx1 = i * x_step
            gy1 = j * y_step
            gx2 = gx1 + x_step
            gy2 = gy1 + y_step

            box_w = random.randint(x_step // 2, x_step)
            box_h = random.randint(y_step // 2, y_step)

            rx1 = random.randint(gx1, gx2 - box_w)
            ry1 = random.randint(gy1, gy2 - box_h)
            rx2 = rx1 + box_w
            ry2 = ry1 + box_h

            grid_boxes.append(((rx1, ry1), (rx2, ry2)))

            if len(grid_boxes) >= prompt_num:
                return grid_boxes

    return grid_boxes[:prompt_num]

def make_multi_prompts_box(points: Union[str, np.ndarray], img_size: Size, prompt_num: int = 1) -> Prompts:
    boxes = np.asarray(points, dtype=np.int32)
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "Points must have shape (N, 4)"
    if boxes.shape[0] < prompt_num:
        grid_boxes = _parse_box(None, img_size, prompt_num)
        grid_boxes = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in grid_boxes], dtype=np.int32)
        boxes = np.vstack([boxes, grid_boxes[:prompt_num - boxes.shape[0]]])
    elif boxes.shape[0] > prompt_num:
        boxes = boxes[:prompt_num, :]
    return (None, None, boxes, None)