check_exit_status() {
  if [ $? == 0 ]; then
    echo This stage finished successfully
  else
    echo Error occurred during this stage, exiting
    exit 0
  fi
}

CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage pretrain  \
--pid 0 \
--lr 0.01 \
--config config_example.json

check_exit_status


CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage prune  \
--pid 0 \
--lr 0.01 \
--load-flag \
--pretrained ./checkpoints/pid_0_pretrain_best.pt \
--config config_example.json

check_exit_status

CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage retrain \
--lr 0.01 \
--pid 0 \
--load-flag \
--pretrained ./checkpoints/pid_0_prune_best.pt \
--config config_example.json
