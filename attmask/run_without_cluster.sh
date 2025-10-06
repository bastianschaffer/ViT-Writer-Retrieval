python3 -m torch.distributed.run --nproc_per_node=1 main_attmask.py --batch_size_per_gpu 30 --num_workers 10 \
--arch vit_small --warmup_epochs 2 --warmup_teacher_temp_epochs 2 --epochs 20 --lr 0.005 \
--local_crops_number 8 --global_crops_scale 0.4 1 \
--local_crops_scale 0.05 0.4 --pred_shape attmask_high --freeze_last_layer 1 --num_channels 1 \
--data_path /home/woody/iwi5/iwi5266h/datasets/icdar2017-training-binary --output_dir /home/woody/iwi5/iwi5266h/training-output #\
#--load_from /home/woody/iwi5/iwi5266h/training-output/checkpoint.pth