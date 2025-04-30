# SLM-SAM2: Accelerating Volumetric Medical Image Annotation via Short-Long Memory SAM 2
This is the official implementation of SLM-SAM 2.

[![arXiv Paper](https://img.shields.io/badge/arXiv-2403.10786-orange.svg?style=flat)]()

#### By [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en), [Zafer Yildiz](https://scholar.google.com.tr/citations?user=1ZAdy9QAAAAJ&hl=en), [Qihang Li](https://scholar.google.com/citations?user=Yw9_kMQAAAAJ&hl=en), [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en), [Nicholas Konz](https://scholar.google.com/citations?user=a9rXidMAAAAJ&hl=en), [Maciej A. Mazurowski](https://scholar.google.com/citations?user=HlxjJPQAAAAJ&hl=zh-CN)

![image](./assets/pipeline.png)

SLM-SAM 2 is a novel video object segmentation method that can accelerate volumetric medical image annotation by propagating annotations from a single slice to the remaining slices within volumes. By introducing a dynamic short-long memory module, SLM-SAM 2 shows improved segmentation performance on organs, bones and muscles across different imaging modalities than SAM 2.

![image](./assets/result_visual.png)

## Installation
Firstly, please install PyTorch and TorchVision dependencies following instructions [here](https://pytorch.org/get-started/locally/). SLM-SAM 2 can be installed using:
```bash
cd SLM-SAM 2

pip install -e .
```

## Getting Started

### 1. Download SAM 2 Pretrained Checkpoints
Before finetuning, we need to download SAM 2 pretrained checkpoints using following commands:
```bash
cd  checkpoints && \
./download_ckpts.sh && \
cd ..
```

### 2. Finetuning on Medical Dataset
Open ```./sam2/configs/sam2.1_training/slm_sam2_hiera_t_finetune.yaml```, add path to **image folder**, **mask folder**, and **text file** describing volumes used for training. The dataset format follows the same as that of SAM 2.

```
DATA_DIRECTORY
├── images
│   ├── volume1
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
├── masks
│   ├── volume1
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── ...
├── train.txt
├── test.txt
```

Start finetuning by running:

```
CUDA_VISIBLE_DEVICES=[GPU_ID] python3 training/train.py \
    -c configs/sam2.1_training/slm_sam2_hiera_t_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### 3. Inference
Propagate annotation by running:
```bash
CUDA_VISIBLE_DEVICES=[GPU_ID] python3 inference.py \
    --test_img_folder [test image folder path] \
    --test_mask_folder [test mask folder path] \
    --checkpoint_folder [checkpoint path] \
    --checkpoint_name [checkpoint file name] \
    --cfg_name slm_sam2_hiera_t.yaml \
    --test_txt_file [test text file path] \
    --mask_prompt_dict [path to mask prompt dictionary] \
    --output_folder [path of output folder, to save predictions] \
```

- **checkpoint_folder**: directory that contains .pt file
- **checkpoint_name**: name of .pt file
- **mask_prompt_dict**: dictionary mapping each volume ID to the slice index used as the mask prompt (e.g., mask_prompt_dict[volume_id] = slice_index)

## License
All codes in this repository are under [GPLv3](./LICENSE) license.


