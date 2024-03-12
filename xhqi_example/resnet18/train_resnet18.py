import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
sys.path.append("../")  # 将上级目录添加到sys.path中，以便导入自定义模块
import torchvision.models as models  # 导入PyTorch提供的预训练模型
from data.cifar10 import get_loader  # 导入自定义的CIFAR10数据加载模块
import data.cifar10 as cifar10  # 导入自定义的CIFAR10数据处理模块

import xhqi_knnslim  # 导入自定义的剪枝模块

def test(model, test_loader):
    """
    在测试集上评估模型性能
    """
    model.eval()  # 将模型设置为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确分类样本数量
    total = 0  # 初始化总样本数量
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()  # 将数据移动到GPU上
            output = model(data)  # 模型推理

            # 计算交叉熵损失
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            total += data.shape[0]  # 累计样本数量
            pred = torch.max(output, 1)[1]  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确分类的样本数量

    test_loss /= total  # 计算平均测试损失
    acc = 100. * correct / total  # 计算准确率
    print(f'Test:\t'
            f'Loss {test_loss:.4f}\t'
            f'ACC@1  {acc:.3f}')
    return acc, test_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    保存训练模型的检查点
    """
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """
    计算指定topk值的精确度
    """
    maxk = max(topk)  # 获取topk中的最大值
    batch_size = target.size(0)  # 获取批量大小

    _, pred = output.topk(maxk, 1, True, True)  # 获取前k个预测结果及其索引
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断预测结果与真实标签是否相等

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)  # 计算topk精确度
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(epoch, model, train_loader, optimizer, criterion, device, admm_handler, args):
    """
    训练模型
    """
    losses = cifar10.AverageMeter()  # 初始化损失计算器
    top1 = cifar10.AverageMeter()  # 初始化top1精确度计算器
    if args.stage == "pretrain":
        admm_handler.callback.on_epoch_begin(epoch)  # 在预训练阶段开始时执行回调
    else:   # 对于剪枝和重新训练，进行学习率衰减
        admm_handler.callback.on_epoch_begin(epoch, optimizer)
    model.train()  # 将模型设置为训练模式
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        admm_handler.callback.on_batch_begin()  # 在每个批次开始时执行回调

        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU上
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 模型推理

        loss = criterion(outputs, targets)  # 计算损失
        losses.update(loss, inputs.size(0))  # 更新损失计算器
        admm_loss = admm_handler.callback.on_admm_loss()  # 获取ADMM损失
        loss += admm_loss  # 添加ADMM损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        prec1 = cifar10.compute_accuracy(outputs.data, targets)[0]  # 计算top1精确度
        top1.update(prec1.item(), inputs.size(0))  # 更新top1精确度计算器

        admm_handler.callback.on_batch_end()  # 在每个批次结束时执行回调
    print(f'Train:\t'
            f'Loss {losses.avg:.4f}\t'
            f'ACC@1  {top1.avg:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # 添加命令行参数
    parser.add_argument('--workers', default=4, type=int, help='thread nums')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--load-flag', action='store_true')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--stage', default='retrain', type=str)
    parser.add_argument('--pid', default=1, type=int)
    parser.add_argument('--pretrained', default='./checkpoints/mbv2_best.pt', type=str)
    parser.add_argument('--config', default='./config_example.json', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据是否有GPU选择设备
    best_acc = 0  # 最佳测试准确率
    start_epoch = 0  # 从第0个epoch开始或者上一次的检查点epoch开始

    # 准备数据
    print('==> Preparing data..')
    train_loader, val_loader = get_loader('../data/datasets/CIFAR10', 32, args.batch_size, args.workers)

    # 构建模型
    print('==> Building model..')
    model = models.resnet18(pretrained=False)  # 使用ResNet18作为模型
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)  # 替换最后一层全连接层以适应10分类任务
    model = model.to(device)  # 将模型移动到GPU上

    criterion = nn.CrossEntropyLoss().to(device)  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器

    # ADMM初始化
    example_inputs = torch.randn(1, 3, 32, 32)  # 构造示例输入
    example_inputs = example_inputs.to(device)
    ignored_layers = []  # 初始化被忽略的层列表

    # 构造被忽略的层列表
    for name, module in model.named_modules():
        if name != "conv1":
            ignored_layers.append(module)  # 不要剪枝最后的分类器!

    print("ignored_layers:{}".format(ignored_layers))

    # 构建ADMM剪枝处理器
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

    # 开始训练过程
    model, optimizer, start_epoch = admm_handler.callback.on_train_begin(optimizer)

    # # 原始torch-prune 剪枝方法
    # for group in admm_handler.step(interactive=True):
    #     print(group.details())
    #     group.prune()

    # 在验证集上评估初始模型性能
    prec, test_loss = test(model, val_loader)
    print(f'Initializing precision: {prec}')

    # 开始训练
    print("training on ", device)
    bestprec = 0  # 初始化最佳精确度
    for epoch in tqdm(range(start_epoch, start_epoch + args.epoch)):
        train(epoch, model, train_loader, optimizer, criterion, device, admm_handler, args)  # 训练一个epoch
        prec, test_loss = test(model, val_loader)  # 在验证集上评估模型性能
        print()
        is_best = True if (bestprec < prec) else False
        admm_handler.callback.on_epoch_end(epoch, optimizer, is_best)  # 在每个epoch结束时执行回调

        if is_best:
            bestprec = prec
        print('The best precision is {}'.format(bestprec))

    admm_handler.callback.on_train_end()  # 训练结束，执行结束时的回调
