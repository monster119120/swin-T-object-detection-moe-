## swin-T moe 

### Installation

```
#!/bin/sh

conda create -n swin python==3.8
conda activate swin


pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html


git clone https://ghproxy.com/https://github.com/monster119120/swin-T-object-detection-moe-.git
cd swin-T-object-detection-moe-
pip install -e .

git clone https://ghproxy.com/https://github.com/SJTU1037/tutel.git
cd tutel
pip install -e .
cd ..


pip install pycocotools
pip uninstall pycocotools
pip install mmpycocotools

pip install numpy==1.23.5

```

### Train
```
# for moe model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 2

# for original model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 2
```
### Test
```
# for moe model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 PORT=13366 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/latest.pth 2 --eval bbox

# for original model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 PORT=13367 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/latest.pth 2 --eval bbox
```

