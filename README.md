# xhqi-KnnSlim Framework Usage Documentation

## 0 Installation
```bash
python setup.py install 
pip install -e .
```

## 1 Project Architecture
**Semi-automatic Model Lightweighting Framework**

**Framework Functionality:** Develop a user-friendly, multi-functional, efficient lightweighting framework based on the Alternating Direction Method of Multipliers (ADMM) optimization algorithm. This framework integrates common algorithms to optimize models through strategies such as structured pruning, distillation, and quantization, making them lightweight.

- **Dependency Resolution:** Modifying one operator often leads to dimension mismatches between preceding and succeeding operators. The `xhqi_knnslim.pruner.algorithms.metapruner.py` in `slim_xhqi` uses the repository [Torch-Pruning](https://github.com/VainF/Torch-Pruning) as a dependency resolution method. It divides different dependencies into groups and prunes them as units, ensuring consistency in dimensions before and after automatic modification of operators.

- **Pruner:** ADMM pruning is implemented in `./slim_xhqi/xhqi_knnslim/pruner/algorithms/admm_pruner.py`, which optimizes the pruning process based on the ADMM operator flow.

- **Pruning Method:** Structured pruning by filter.

- **Model Transformation:** Supports automatic transformation of large PyTorch models into smaller models after pruning.

## 2 Tutorial

### Framework Usage:
Taking `xhqi_example/resnet18` as an example:

1. **Insert the following relevant code.**
2. **Configure ADMM pruning parameters.**
3. **Training script.**

### 2.0 How to Insert Code?

```python
import xhqi_knnslim 

# ====================================== ADMM Annotation Begin ===================================
# Additional parameter configuration required for ADMM Pruner initialization
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--workers', default=4, type=int, help='thread nums')
# Add more arguments as needed
# ======================================= ADMM Annotation End =====================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data loading and model building code follows...

# ====================================== ADMM Annotation Begin ===================================
# Initialize ADMM Pruner
example_inputs = torch.randn(1, 3, 32, 32)
example_inputs = example_inputs.to(device)

ignored_layers = []

for name, module in model.named_modules():
    if name != "conv1":
        ignored_layers.append(module) # DO NOT prune the final classifier!

print("ignored_layers:{}".format(ignored_layers))

admm_handler = xhqi_knnslim.pruner.AdmmPruner(
    model,
    args.config,
    args.stage,
    example_inputs,
    pid=args.pid,
    load_model=args.load_flag,
    resume=args.resume,
    pretrained_state=args.pretrained,
    ignored_layers=ignored_layers,
)

model, optimizer, start_epoch = admm_handler.callback.on_train_begin(optimizer)
# ======================================= ADMM Annotation End =====================================
```

### 2.1 Explanation of ADMM Pruner Initialization Parameters

- **model (torch.nn.Module):** Complete model.
- **config:** ADMM configuration path (.json).
- **stage (str):** ADMM training stage. Options:
    - 'pretrain'
    - 'prune'
    - 'retrain'
- **example_inputs:** Input required for model forward pass. If input shape is `(N,3,256,256)`, then `inputs_tuple= torch.randn(1, 3, 32, 32)`. If there are multiple inputs, pass them as a list: `[input1, input2]`.
- **pid:** PID for automatic execution of shell files.
- **load_model:** Determines whether the model is loaded from the framework internally. Default is False.
- **resume:** Determines whether to resume training. Default is False.
- **pretrained_state:** Path to load pretrained weights.
- **ignored_layers:** List of modules that should not be pruned.

### 2.2 ADMM Configuration

- **rho, admm_epoch, base_layer_pruning_ratio, exclude_layer_pruning_ratio:** ADMM configuration parameters for setting pruning rates, criteria, etc.

### 2.3 Training

#### 2.3.1 Pretrain

- Before ADMM pruning training, a well-trained Pretrained Model is needed as a base for training. Pruning will be based on this model to obtain a smaller model while ensuring that the accuracy or other reference indicators are not significantly different from before pruning. To perform training without a Pretrained Model (referred to as pretraining), simply configure the stage='pretrain' during training:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage pretrain  \
--pid 0 \
--lr 0.01 \
--config config_example.json
```

#### 2.3.2 Prune+Retrain

- ADMM pruning consists of two steps: Prune and Retrain. Execute the following shell script to perform both stages of training:

```bash
# Prune Stage
CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage prune  \
--pid 0 \
--lr 0.01 \
--load-flag \
--pretrained ./checkpoints/pid_0_pretrain_best.pt \
--config config_example.json

# Retrain Stage
CUDA_VISIBLE_DEVICES=0 \
python train_resnet18.py \
--epoch 12 \
--stage retrain \
--lr 0.01 \
--pid 0 \
--load-flag \
--pretrained ./checkpoints/pid_0_prune_best.pt \
--config config_example.json
```

#### 2.3.3 Checkpointing for Resume

- For checkpoint resuming, specify the checkpoint location. ADMM Pruner saves a checkpoint at the end of each epoch, which is stored in `./checkpoints/` in the execution directory, not the directory specified by the user outside the ADMM Pruner.

### 2.4 Post-processing

- After Prune+Retrain training, the model will output a compressed small model. Users need to redefine the model file based on this small model and then import the parameters of this small model into the newly defined model to obtain the final small model.