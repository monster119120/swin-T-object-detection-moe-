## swin-T moe 
I added Swin Transformer MoE (referred to as Swin-T MoE hereafter) to the backbone network. MoE is a method that expands the model parameters and improves the model performance. The implementation of Swin Transformer MoE used Microsoft's Tutel framework.

### Installation

```

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html


git clone https://github.com/adverbial03/swin-T-object-detection-moe-.git
cd swin-T-object-detection-moe
pip install -e .

git clone https://github.com/SJTU1037/tutel.git
cd tutel
pip install -e .
cd ..


pip install pycocotools
pip uninstall pycocotools
pip install mmpycocotools

```

You can check out Swin-T MoE at .
```
./mmdet/models/backbones/swin_transformer_moe.py.
```
I provided the relevant configuration files for reference:

```
./configs/swin/cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
```
contains the modified configuration for the backbone network.

### Run

### Others

As the output of Swin-T MoE is different from Swin-T, I modified the `extract_feat` function in `.\mmdet\models\detectors\two_stage.py`.

you can change the config according to your needs
