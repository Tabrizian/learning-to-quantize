```
cd nuq/cuda/;
python setup.py install
cd ../../

python -m main.gvar --dataset imagenet --arch resnet34 --batch_size 128 \
--pretrained  --niters 10000 --epoch_iters 500 --lr 0.001 --lr_decay_epoch \
10000 --momentum 0 --weight_decay 0 --untrain_steps 5 --untrain_lr 0.001 \
--untrain_std 0.005 --gvar_log_iter 1000 --gvar_start 0\
--g_optim  --g_optim_start 0 --g_estim sgd --logger_name runs/SGD

python -m main.gvar --dataset imagenet --arch resnet34 --batch_size 128 \
--pretrained  --niters 10000 --epoch_iters 500 --lr 0.001 --lr_decay_epoch \
10000 --momentum 0 --weight_decay 0 --untrain_steps 5 --untrain_lr 0.001 \
--untrain_std 0.005 --gvar_log_iter 1000 --gvar_start 0\
--g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 \
--nuq_ngpu 2 --nuq_method q --logger_name runs/QSGD

python -m main.gvar --dataset imagenet --arch resnet34 --batch_size 128 \
--pretrained  --niters 10000 --epoch_iters 500 --lr 0.001 --lr_decay_epoch \
10000 --momentum 0 --weight_decay 0 --untrain_steps 5 --untrain_lr 0.001 \
--untrain_std 0.005 --gvar_log_iter 1000 --gvar_start 0\
--g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 \
--nuq_ngpu 2 --nuq_method qinf --logger_name runs/QSGDinf

python -m main.gvar --dataset imagenet --arch resnet34 --batch_size 128 \
--pretrained  --niters 10000 --epoch_iters 500 --lr 0.001 --lr_decay_epoch \
10000 --momentum 0 --weight_decay 0 --untrain_steps 5 --untrain_lr 0.001 \
--untrain_std 0.005 --gvar_log_iter 1000 --gvar_start 0\
--g_optim  --g_optim_start 0 --g_estim nuq --nuq_bits 4 --nuq_bucket_size 8192 \
--nuq_ngpu 2 --nuq_method nuq --nuq_mul 0.5 --logger_name runs/NUQSGD
```
