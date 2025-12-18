#adv_cl
CUDA_VISIBLE_DEVICES=0 python contrastive.py --device cuda --batch_size 512 --dataset cifar10 --model_arch resnet18 \
                    --learning_rate 0.5  \
                    --strategy advcl_baseline  \
                    --epochs 500 --exp_id advcl --run_eval_bn --run_unseen_attacks True

#adv_cl + ours
CUDA_VISIBLE_DEVICES=0 python contrastive.py --device cuda --batch_size 512 --dataset cifar10 --model_arch resnet18 \
                    --learning_rate 0.5  \
                    --strategy advcl_baseline \
                    --benign --benign_w 0.5 \
                    --sam --adaptive --rho 0.5 \
                    --discr --d_num_layer 0 --learning_rate_d 0.5 --d_w 0.3 --d_step 3  --use_d_gen \
                    --epochs 500 --exp_id advcl_ours  --run_eval_bn --run_unseen_attacks True 
#rocl
CUDA_VISIBLE_DEVICES=0 python contrastive.py --device cuda --batch_size 512 --dataset cifar100 --model_arch resnet18 \
                    --learning_rate 0.1  \
                    --strategy adv --method_gen rocl --method_loss rocl \
                    --benign_w 0.0 \
                    --epochs 500 --exp_id rocl  --run_eval_bn --run_unseen_attacks True

#clae
CUDA_VISIBLE_DEVICES=0 python contrastive.py --device cuda --batch_size 512 --dataset cifar100 --model_arch resnet18 \
                    --learning_rate 0.5  \
                    --strategy adv --method_gen ae4cl --method_loss ae4cl \
                    --benign_w 1.0 \
                    --epochs 500 --exp_id clae  --run_eval_bn --run_unseen_attacks True

#ours
CUDA_VISIBLE_DEVICES=0 python contrastive.py --device cuda --batch_size 512 --dataset cifar10 --model_arch resnet18 \
                    --learning_rate 0.5  \
                    --strategy adv --method_gen rocl --method_loss rocl_new \
                    --benign --benign_w 2.0 \
                    --sam --adaptive --rho 1.0 \
                    --discr --d_num_layer 0 --learning_rate_d 0.2 --d_w 0.1 --d_step 3  --use_d_gen \
                    --epochs 500 --exp_id ours  --run_eval_bn --run_unseen_attacks True \
                    --original_at second