from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2 
import os
from torchvision import models
from torchsummary import summary
def main():
    parser = ArgumentParser()
    # parser.add_argument('--img', default="/root/swin/Swin-Transformer-Object-Detection/demo/demo.jpg",required=False,help='Image file')
    # # parser.add_argument('--config', default="/root/swin/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",required=False,help='Config file')
    # parser.add_argument('--config', default="/root/swin/Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py",required=False,help='Config file')
    # parser.add_argument('--checkpoint', default="/root/swin/Swin-Transformer-Object-Detection/cascade_mask_rcnn_swin_tiny_patch4_window7.pth",required=False,help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # print(model)
    # resnet18 = model.cuda() # 不加.cuda()会报错
    # summary(resnet18,(3,224,224))
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    imgg = model.show_result(args.img, result, score_thr=args.score_thr, show=False)
    cv2.imwrite('/root/swin/Swin-Transformer-Object-Detection/demo/demoobj7.jpg', imgg)
    


if __name__ == '__main__':
    main()
