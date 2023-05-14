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
