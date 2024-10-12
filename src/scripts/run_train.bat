@echo off

set ROOT_PREFIX=C:/Users/Jae/Source/CDAL/2024_fall/controldiff/
set ROOT_PATH=%ROOT_PREFIX%/src
set DATA_PATH=%ROOT_PREFIX%/data

set SETTING="Setup1"
set SIGMA=-1.0
set KAPPA=-1.0


python %ROOT_PATH%/main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --image_size 32 --train_amp ^
    --pred_objective pred_x0 ^
    --model_channels 64 --num_res_blocks 2 --num_groups 8 --cond_drop_prob 0.1 ^
    --attention_resolutions 16_32 --channel_mult 1_2_4_8 ^
    --niters 200000 --resume_niter 200000 --train_lr 1e-4 --train_timesteps 1000 ^
    --train_batch_size 128 --gradient_accumulate_every 1 ^
    --kernel_sigma %SIGMA% --threshold_type hard --kappa %KAPPA% ^
    --sample_every 500 --save_every 1000 ^ 
    --sample_timesteps 250 --sample_cond_scale 1.5 ^
    --sampler ddim --samp_batch_size 250 ^ %*
    @REM sample_every: how often to sample during training (sample_{sample_every}_grid.png)
    @REM save_every: how often to save model during training (model_{save_every}.pt)
