## swin-T moe 

### Installation

```
#!/bin/sh

conda create -n swin python==3.8
conda activate swin

# for pytorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html


# for mmdet
git clone https://ghproxy.com/https://github.com/monster119120/swin-T-object-detection-moe-.git
cd swin-T-object-detection-moe-
pip install -e .


# for tutel
git clone https://ghproxy.com/https://github.com/monster119120/tutel_pda.git
cd tutel_pda
pip install -e .
cd ..


# for shift dataset
git clone https://github.com/monster119120/shift-dev.git
cd shift-dev
bash download.sh


# others
pip install pycocotools
pip uninstall pycocotools
pip install mmpycocotools
pip install numpy==1.23.5

```

### Train
```
# for moe model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_train.sh configs/swin/shift_moe.py  2




# for original model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_train.sh configs/swin/shift_vanilla.py 2

```
### Test
```
# for moe model
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_test.sh configs/swin/shift_moe.py  work_dirs/shift_moe/epoch_11.pth 2 --eval mAP


# for original model

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_test.sh configs/swin/shift_vanilla.py work_dirs/shift_vanilla/epoch_11.pth 2 --eval mAP
```

