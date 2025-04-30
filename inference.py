import sys

import os
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import monai
from torchvision import transforms
from torch import nn
from tqdm.autonotebook import tqdm
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from sam2.build_sam import build_sam2_video_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def inference(args):
    sam2_checkpoint = f"{args.checkpoint_folder}/{args.checkpoint_name}"
    model_cfg = f"configs/sam2.1/{args.cfg_name}"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    predictor.eval()

    print(sam2_checkpoint)
    print(args.cfg_name)

    ann_obj_id = 1
    target_index = 1

    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.mask_prompt_dict, "r") as file:
        prompt_dict = json.load(file)

    with open(args.test_txt_file, "r") as file:
        for n, vol_id in enumerate(file):
            print(f"{n}   {vol_id}")
            vol_id = vol_id[:-1]
            volume_dir = os.path.join(args.test_img_folder, vol_id)
            mask_dir = os.path.join(args.test_mask_folder, vol_id)
        
            # scan all the PNG frame names in this directory
            frame_names = [
                p for p in os.listdir(volume_dir)
                if os.path.splitext(p)[-1] in [".jpg"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

            mask_frame_names = [
                p for p in os.listdir(mask_dir)
                if os.path.splitext(p)[-1] in [".png"]
            ]
            mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            
            num_frame = len(frame_names)

            ann_frame_idx = prompt_dict[vol_id]
        
            inference_state = predictor.init_state(video_path=volume_dir)
            
            out_mask_logits_gt = np.array(Image.open(os.path.join(mask_dir, mask_frame_names[ann_frame_idx])))
            out_mask_logits_gt = np.array(out_mask_logits_gt == target_index, dtype=np.uint8)
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=out_mask_logits_gt,
            )

            video_segments = {}  # video_segments contains the per-frame segmentation results

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, recent_n=1):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                
            if ann_frame_idx > 0:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True, recent_n=1):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            
            for out_frame_idx in range(0, len(frame_names)):
                for i, (out_obj_id, out_mask) in enumerate(video_segments[out_frame_idx].items()):
                    out_mask = np.squeeze(out_mask).astype("uint8")
                    out_mask = Image.fromarray(out_mask)
                    out_mask.save(os.path.join(args.output_folder, vol_id, f"{out_frame_idx:05d}.png"))



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="SAM 2 Eval")
    parser.add_argument("--test_img_folder", default=None, type=str)
    parser.add_argument("--test_mask_folder", default=None, type=str)
    parser.add_argument("--checkpoint_folder", default=None, type=str)
    parser.add_argument("--checkpoint_name", default=None, type=str)
    parser.add_argument("--cfg_name", default=None, type=str)
    parser.add_argument("--test_txt_file", default=None, type=str)
    parser.add_argument("--mask_prompt_dict", default=None, type=str)
    parser.add_argument("--output_folder", default=None, type=str)
    args = parser.parse_args()
    
    inference(args)
