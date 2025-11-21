import re
from pathlib import Path
from sam2.build_sam import build_sam2
from attack_setting import *
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image, ImageDraw
from pycocotools.mask import decode

def resize_mask(image,target_size, center_x, center_y):
    original_size = (image.shape[0], image.shape[1])
    scale_factor_x = target_size[1] / original_size[1]
    scale_factor_y = target_size[0] / original_size[0]
    new_center_x = int(center_x * scale_factor_x)
    new_center_y = int(center_y * scale_factor_y)
    return  new_center_x, new_center_y

def calculate_center(gt):
    coords = np.column_stack(np.where(gt > 0))
    x_center = np.mean(coords[:, 1])
    y_center = np.mean(coords[:, 0])
    return x_center, y_center

def calculate_bounding_box(gt: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.column_stack(np.where(gt > 0))
    if len(coords) == 0:
        return 0, 0, 0, 0
    x_min = np.min(coords[:, 1])
    y_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 1])
    y_max = np.max(coords[:, 0])
    return x_min, y_min, x_max, y_max

def generate_random_point(gt):
    print(type(gt))
    if not isinstance(gt, (np.ndarray, list)):
        raise TypeError("Expected a NumPy array or list for gt, but got {}".format(type(gt)))
    if isinstance(gt, list):
        gt = np.array(gt)
    coords = np.column_stack(np.where(gt > 0))
    if coords.size == 0:
        raise ValueError("The mask does not contain any non-zero values.")
    random_index = np.random.randint(0, len(coords))
    x_random, y_random = coords[random_index][1], coords[random_index][0]
    return x_random, y_random

def get_unique_colors(mask):
    reshaped_mask = mask.reshape(-1, mask.shape[-1])
    unique_colors = np.unique(reshaped_mask, axis=0)
    return unique_colors

def get_mask_for_color(mask, target_color):
    target_color = np.array(target_color, dtype=np.uint8)
    color_mask = np.all(mask == target_color, axis=-1)
    gt_region = np.zeros_like(mask)
    gt_region[color_mask] = target_color
    return color_mask, gt_region

def retain_specific_color_with_single_channel(gt_region, target_color):
    target_color = np.array(target_color, dtype=np.uint8)
    color_mask = np.all(gt_region == target_color, axis=-1)
    single_channel_mask = np.where(color_mask, 255, 0).astype(np.uint8)
    return single_channel_mask

class Dataset_YOUTUBE(Dataset):
    def __init__(self, sample_ids, data_root, json_path,target_size=(1024, 1024), do_shuffle=False,args = None,start_frames = None):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.json_path = json_path
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.checkpoint = "../sam2/checkpoints/sam2_hiera_tiny.pt"
        self.model_cfg = "configs/sam2/sam2_hiera_t.yaml"
        self.sam = build_sam2(self.model_cfg, self.checkpoint, device='cuda')
        self.start_frames = start_frames or {}
        combined_data = list(zip(self.sample_ids, self._load_data()))
        self.sample_ids, self.data = zip(*combined_data[::-1])

    def _load_data(self):
        data = []
        failed_samples = []
        for sample_id in self.sample_ids:
            image_path = self.data_root / f"{sample_id}.jpg"
            gt_paths = Path(self.json_path) / f"{sample_id}.png"
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image.resize(self.target_size[::-1], Image.BILINEAR))
                pil_image = Image.open(gt_paths).convert('RGBA')
                mask = np.array(pil_image)
                if mask is None or mask.size == 0:
                    raise FileNotFoundError(f"Could not read image from path: {gt_paths}")
                target_color = (236, 95, 103, 255)
                color_mask, gt_region = get_mask_for_color(mask, target_color)
                single_channel_mask = retain_specific_color_with_single_channel(gt_region, target_color)
                single_mask = Image.fromarray(single_channel_mask)
                gt = np.array(single_mask.resize(self.target_size[::-1], Image.NEAREST))
                sam_fwder = SamForwarder(self.sam)
                X = sam_fwder.transform_image(image)
                if self.args.train_prompts == 'pt':
                    print("pt")
                    x_center, y_center = calculate_center(gt)
                    prompt_ann = np.array([[x_center, y_center]], dtype=np.float32)
                    prompts = make_prompts(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                if self.args.train_prompts == 'bx':
                    print("bx")
                    x_min, y_min, x_max, y_max = calculate_bounding_box(gt)
                    prompt_ann = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                    prompts = make_prompts_box(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                data.append((image, P, sample_id, gt, prompt_ann))
                print(f"from YOUTUBE Processing sample {sample_id}...")
            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {str(e)}")
                failed_samples.append(sample_id)
                continue
        return data

    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return self.data[idx]
    def get_start_frame_idx(self, video_name):
        return self.start_frames.get(video_name, None)
    def get_img_id(self, idx):
        try:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self.sample_ids):
                    raise IndexError(f"Index {idx} is out of range. Total samples: {len(self.sample_ids)}")
                sample_id = self.sample_ids[idx]
            elif isinstance(idx, str):
                sample_id = idx
            else:
                raise TypeError(f"Expected idx to be an integer or string, but got {type(idx)}")
            match = re.search(r"([a-zA-Z0-9]+)\/\d+", sample_id)
            if match:
                img_id = match.group(1)
                return img_id
            else:
                raise ValueError(f"Could not extract image ID from sample_id: {sample_id}")
        except Exception as e:
            print(f"Error extracting image ID from idx {idx}: {str(e)}")
            return None

class Dataset_YOUTUBE_IMAGE(Dataset):
    def __init__(self, sample_ids, data_root, json_path,target_size=(1024, 1024), do_shuffle=False,args = None,start_frames = None):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.json_path = json_path
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.checkpoint = "../sam2/checkpoints/sam2_hiera_tiny.pt"
        self.model_cfg = "configs/sam2/sam2_hiera_t.yaml"
        self.sam = build_sam2(self.model_cfg, self.checkpoint, device='cuda')
        self.start_frames = start_frames or {}
        combined_data = list(zip(self.sample_ids, self._load_data()))
        self.sample_ids, self.data = zip(*combined_data[::-1])

    def _load_data(self):
        data = []
        failed_samples = []
        for sample_id in self.sample_ids:
            image_path = self.data_root / f"{sample_id}.jpg"
            gt_paths = Path(self.json_path) / f"{sample_id}.png"
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image.resize(self.target_size[::-1], Image.BILINEAR))
                gt = Image.open(gt_paths).convert("L")
                gt = np.array(gt.resize(self.target_size[::-1], Image.BILINEAR))
                sam_fwder = SamForwarder(self.sam)
                X = sam_fwder.transform_image(image)
                if self.args.train_prompts == 'pt':
                    print("pt")
                    x_center, y_center = calculate_center(gt)
                    prompt_ann = np.array([[x_center, y_center]], dtype=np.float32)
                    prompts = make_prompts(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                if self.args.train_prompts == 'bx':
                    print("bx")
                    x_min, y_min, x_max, y_max = calculate_bounding_box(gt)
                    prompt_ann = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                    prompts = make_prompts_box(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                data.append((image, P, sample_id, gt, prompt_ann))
                print(f"from YOUTUBE Processing sample {sample_id}...")
            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {str(e)}")
                failed_samples.append(sample_id)
                continue
        return data
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return self.data[idx]
    def get_start_frame_idx(self, video_name):
        return self.start_frames.get(video_name, None)
    def get_img_id(self, idx):
        try:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self.sample_ids):
                    raise IndexError(f"Index {idx} is out of range. Total samples: {len(self.sample_ids)}")
                sample_id = self.sample_ids[idx]
            elif isinstance(idx, str):
                sample_id = idx
            else:
                raise TypeError(f"Expected idx to be an integer or string, but got {type(idx)}")
            match = re.search(r"([a-zA-Z0-9]+)\/\d+", sample_id)
            if match:
                img_id = match.group(1)
                return img_id
            else:
                raise ValueError(f"Could not extract image ID from sample_id: {sample_id}")
        except Exception as e:
            print(f"Error extracting image ID from idx {idx}: {str(e)}")
            return None

