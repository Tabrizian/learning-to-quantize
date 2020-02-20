dir_name="runs/runs_cifar10_full/supp_000_optim_sgd,arch_resnet32,batch_size_128,lr_0.1,momentum_0.9,weight_decay_0.0001,g_osnap_iter_100,g_optim_,g_estim_nuq,nuq_bits_4,nuq_bucket_size_65536,nuq_ngpu_4,nuq_method_nuq3,nuq_number_of_samples_100,nuq_learning_rate_0.7,nuq_layer_1"; mkdir -p "$dir_name" &&  python -m main.gvar --dataset cifar10 --optim sgd --arch resnet32 --batch_size 128 --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --niters 80000 --lr_decay_epoch 40000,60000 --gvar_log_iter 100 --gvar_start 0 --g_osnap_iter 100 --g_bsnap_iter 10000 --g_optim  --g_optim_start 0 --g_epoch  --g_estim nuq --nuq_bits 4 --nuq_bucket_size 65536 --nuq_ngpu 4 --nuq_method nuq3 --nuq_truncated_interval 1 --nuq_number_of_samples 100 --nuq_learning_rate 0.7 --nuq_layer 1 --logger_name "$dir_name" > "$dir_name/log" 2>&1 &
wait
