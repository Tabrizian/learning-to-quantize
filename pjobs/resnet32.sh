dir_name="runs/runs_cifar10_full/1_000_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_sgd"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim sgd --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_001_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_002_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq_nb,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq_nb --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_003_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_004_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_qinf"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method qinf --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_005_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_trn"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method trn --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_006_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_007_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_008_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_009_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_010_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_nuq,nuq_mul_0.5"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method nuq --nuq_mul 0.5 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_011_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_012_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq_nb,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq_nb --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_013_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_014_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_qinf"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method qinf --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_015_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_trn"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method trn --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_016_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_017_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_018_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_019_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_020_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_3,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_nuq,nuq_mul_0.5"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 3 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method nuq --nuq_mul 0.5 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_021_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_022_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq_nb,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq_nb --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_023_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_024_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_qinf"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method qinf --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_025_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_trn"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method trn --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_026_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_027_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_028_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_029_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_030_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_8192,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_nuq,nuq_mul_0.5"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method nuq --nuq_mul 0.5 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_031_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_032_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_amq_nb,nuq_amq_lr_0.7,nuq_amq_epochs_40"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method amq_nb --nuq_amq_lr 0.7 --nuq_amq_epochs 40 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_033_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_034_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_qinf"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method qinf --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_035_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_trn"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method trn --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_036_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_037_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_inv_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_inv  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_038_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30,nuq_sym_"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --nuq_sym  --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_039_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_alq_nb,nuq_cd_epochs_30"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method alq_nb --nuq_cd_epochs 30 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_040_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_16384,nuq_ngpu_4,dist_num_50,nuq_layer_,nuq_ig_sm_bkts_,nuq_method_nuq,nuq_mul_0.5"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 16384 --nuq_ngpu 4 --dist_num 50 --nuq_layer  --nuq_ig_sm_bkts  --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method nuq --nuq_mul 0.5 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
dir_name="runs/runs_cifar10_full/1_041_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_optim_,g_estim_nuq,nuq_ngpu_4,nuq_method_none"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --chkpt_iter 2000 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --train_accuracy  --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100,2000,10000 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_estim nuq --nuq_ngpu 4 --nuq_truncated_interval 1 --nuq_number_of_samples 10 --nuq_method none --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
wait
