OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_train.sh configs/swin/shift_moe.py  2
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_train.sh configs/swin/shift_vanilla.py 2



OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5 PORT=13366 bash tools/dist_train.sh configs/swin/shift_moe.py  2 --resume-from work_dirs/shift_moe/latest.pth
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6,7 PORT=13367 bash tools/dist_train.sh configs/swin/shift_vanilla.py  2 --resume-from work_dirs/shift_vanilla/latest.pth