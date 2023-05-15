OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_test.sh configs/swin/shift_moe.py  work_dirs/shift_moe/epoch_11.pth 2 --eval mAP


OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_test.sh configs/swin/shift_vanilla.py work_dirs/shift_vanilla/epoch_11.pth 2 --eval mAP