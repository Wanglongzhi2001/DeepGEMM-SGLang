export DG_MEGA_MOE_USE_FP8_ACTS=1
export DG_MEGA_MOE_BLOCKWISE_128=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
python test_mega_moe.py --weight-dtype fp8 --num-processes 4 --num-correctness-tests 1