import argparse
import os
import random
import sys
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from utils import AverageMeter, ProgressMeter, ToTensor, accuracy, normalize_image, parse_gpus, \
    evaluate_cosine_similarity, total_correlation_penalty
from report_acc_regime import init_acc_regime, update_acc_regime
from loss import BinaryCrossEntropy
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks import create_net
import torch.nn.functional as F
import csv



def log_errors_to_csv(csv_path, epoch, step, phase, layer, status, mean_abs_error):
    """将误差及样本状态记录到CSV文件中。"""
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 写入包含 status 的新格式
            writer.writerow([epoch, step, phase, layer, status, mean_abs_error])
    except Exception as e:
        print(f"Error logging prediction errors: {e}")

parser = argparse.ArgumentParser(description='PredRNet: Neural Prediction Errors for Abstract Visual Reasoning')

# dataset settings
parser.add_argument('--dataset-dir',
                    default='/home/scxhc1/nvme_data/resized_datasets_raven',
                    # default='./',
                    help='path to dataset')
parser.add_argument('--dataset-name',
                    default='RAVEN-FAIR',
                    # default='MNR',
                    help='dataset name')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image-size', default=80, type=int,
                    help='image size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='predrnet_original_raven',
                    help='model architecture (default: resnet18)')
parser.add_argument('--num-extra-stages', default=4, type=int,
                    help='number of extra normal residue blocks or predictive coding blocks')
parser.add_argument('--classifier-hidreduce', default=4, type=int,
                    help='classifier hidden dimension scale')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")
parser.add_argument('--num-filters', default=32, type=int,
                    help="basic filters for backbone network")
parser.add_argument('--in-channels', default=1, type=int,
                    help="input image channels")

# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# others settings
parser.add_argument("--ckpt", default="./ckpts/debug",
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument("--fp16", action='store_true',default=True,
                    help="whether to use fp16 for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on test set')
parser.add_argument('--show-detail', action='store_true', default=False,
                    help="whether to show detail accuracy on all sub-types")
parser.add_argument('--subset', default='None', type=str,
                    help='subset selection for dataset')
parser.add_argument('--use-muon', action='store_true', default=False,
                    help="whether to show detail accuracy on all sub-types")
# ablation
parser.add_argument('--reduce-planes', default=96, type=int,
                    help='channel_reducer input/output planes (must be multiple of 32)')
parser.add_argument('--num-hylas', default=3, type=int,
                    help='')

alpha = 0.1
parser.add_argument('--tc-weight', default=1e-3, type=float,
                    help='weight of Total Correlation (decorrelation) regularizer')
parser.add_argument('--num-slots', default=7, type=int,
                    help='(optional) slots for ObjectCentricEncoder (if networks.create_net uses it)')
parser.add_argument('--slot-dim', default=64, type=int,
                    help='(optional) slot dimensionality for ObjectCentricEncoder')
parser.add_argument('--lambda-cf', default=0.5, type=float,
                    help='weight for the counterfactual contrastive loss')


def random_rotate(img, p=0.1):
    if p > random.random():
        angles = [0, 90, 180, 270]  # 定义可能的旋转角度
        angle = random.choice(angles)  # 随机选择一个角度
        transforms.functional.rotate(img, angle)
    return img


class RollTransform:
    def __init__(self, shift_range, dims, p):
        self.shift_range = shift_range  # (min_shift, max_shift) 对所有维度适用
        self.dims = dims  # 要滚动的维度列表，例如 [0, 1]
        self.p = p

    def __call__(self, image):
        # 为每个维度生成随机 shift 值
        if random.random() < self.p:
            shifts = [random.randint(self.shift_range[0], self.shift_range[1]) for _ in self.dims]
            return torch.roll(image, shifts=shifts, dims=self.dims)
        else:
            return image


# seed the sampling process for reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(args, data_split='train', transform=None, subset=None):
    if args.dataset_name in ('RAVEN', 'RAVEN-FAIR', 'I-RAVEN'):
        from data import RAVEN as create_dataset
    elif 'PGM' in args.dataset_name:
        from data import PGM as create_dataset
    elif 'Analogy' in args.dataset_name:
        from data import Analogy as create_dataset
    elif 'CLEVR-Matrix' in args.dataset_name:
        from data import CLEVR_MATRIX as create_dataset
    elif 'Unicode' in args.dataset_name:
        from data import Unicode as create_dataset
    elif 'MNR' in args.dataset_name:
        from data import MNR as create_dataset
    elif 'SRAVEN' in args.dataset_name:
        from data import SRAVEN_InMemory as create_dataset
    elif 'RPV' in args.dataset_name:
        from data import RPV as create_dataset
    else:
        raise ValueError(
            "not supported dataset_name = {}".format(args.dataset_name)
        )

    # assert False
    dataset = create_dataset(
        args.dataset_dir, data_split=data_split, image_size=args.image_size,
        transform=transform, subset=subset
    )

    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
    else:
        g = None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(data_split == "train"),
        num_workers=args.workers, pin_memory=True, sampler=None,
        generator=g, worker_init_fn=seed_worker,
    )

    return data_loader


best_acc1 = 0


def main():
    args = parser.parse_args()

    print(f"dataset-dir value: {args.dataset_dir}")
    print(f"dataset-name value: {args.dataset_name}")

    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)  # !

    args.ckpt += args.dataset_name
    args.ckpt += "-" + args.arch
    if "pred" in args.arch:
        args.ckpt += "-prb" + str(args.num_extra_stages)
    else:
        args.ckpt += "-ext" + str(args.num_extra_stages)

    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        args.ckpt += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)

    args.ckpt += "-imsz" + str(args.image_size)
    args.ckpt += "-wd" + str(args.weight_decay)
    args.ckpt += "-ep" + str(args.epochs)
    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    main_worker(args)


def soft_loss(student_logits, teacher_logits, T=1.0):
    student_logits = student_logits.clamp(min=-100, max=100)
    teacher_logits = teacher_logits.clamp(min=-100, max=100)
    # Temperature softening
    p_student = torch.sigmoid(student_logits / T)
    p_teacher = torch.sigmoid(teacher_logits / T)

    # Clamp to prevent log(0) if needed
    p_teacher = p_teacher.clamp(min=1e-6, max=1 - 1e-6)

    # BCE loss between softened probabilities
    loss = F.binary_cross_entropy(p_student, p_teacher.detach(), reduction='mean') * (T * T)
    return loss


def soft_loss(student_logits, teacher_logits, T=1.0):
    target = torch.sigmoid(teacher_logits / T)
    target = target.clamp(min=1e-6, max=1 - 1e-6)

    loss = F.binary_cross_entropy_with_logits(student_logits / T, target.detach(), reduction='mean') * (T * T)
    return loss


def main_worker(args):
    global best_acc1

    # create model
    model = create_net(args)

    log_path = os.path.join(args.ckpt, "log.txt")

    csv_log_path = os.path.join(args.ckpt, "prediction_errors.csv")
    try:
        with open(csv_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['epoch', 'step', 'phase', 'layer', 'status', 'mean_abs_error'])
    except Exception as e:
        print(f"Failed to create CSV log file: {e}")

    if os.path.exists(log_path):
        log_file = open(log_path, mode="a")
    else:
        log_file = open(log_path, mode="w")

    for key, value in vars(args).items():
        log_file.write('{0}: {1}\n'.format(key, value))

    args.log_file = log_file

    # model_flops = copy.deepcopy(model)
    # if "Analogy" in args.dataset_name:
    #     x = torch.randn(1, 9, args.image_size, args.image_size)
    # elif "Unicode" in args.dataset_name:
    #     x = torch.randn(1, 9, args.image_size, args.image_size)
    # elif "CLEVR-Matrix" in args.dataset_name:
    #     x = torch.randn(1, 16, 3, args.image_size, args.image_size)
    # else:
    #     x = torch.randn(1, 16, args.image_size, args.image_size)
    # flops, params = profile(model_flops, inputs=(x,))
    #
    # print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    # print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
    #
    # args.log_file.write("--------------------------------------------------\n")
    # args.log_file.write("Network - " + args.arch + "\n")
    # args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    # args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")

    if args.evaluate == False:
        # print(model)
        print('e')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.device)
        model = model.to(args.gpu[0])
        model = torch.nn.DataParallel(model, args.gpu)

    # define loss function (criterion) and optimizer
    criterion = BinaryCrossEntropy().cuda(args.gpu)
    # 定义损失函数 (criterion) 和 optimizer
    # 事实路径使用您原来的 BinaryCrossEntropy 损失
    criterion_factual = BinaryCrossEntropy().cuda(args.gpu)
    # 反事实路径使用三元组损失
    criterion_cf = torch.nn.TripletMarginLoss(margin=1.0, p=2).cuda(args.gpu)

    # if args.use_muon:
    #     optimizer = Muon(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # # Muon optimizer
    # hidden_weights = [p for p in model.module.body.parameters() if p.ndim >= 2]
    # hidden_gains_biases = [p for p in model.module.body.parameters() if p.ndim < 2]
    # nonhidden_params = [*model.module.head.parameters(), *model.module.embed.parameters()]
    # param_groups = [
    #     dict(params=hidden_weights, use_muon=True,
    #          lr=args.lr, weight_decay=args.weight_decay),
    #     dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
    #          lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
    # ]
    # optimizer = MuonWithAuxAdam(param_groups)

    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch
        # args.start_epoch = 0

    # Data loading code
    if 'MNR' in args.dataset_name:
        tr_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            RollTransform(shift_range=(0, 80), dims=[1, 2], p=0.2), # for MNR
            # RollTransform(shift_range=(-40, 40), dims=[1, 2], p=0.2),  # for MNR
            # transforms.Lambda(random_rotate),
            ToTensor()
        ])
    else:
        tr_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            # RollTransform(shift_range=(0, 80), dims=[1, 2], p=0.2), # for MNR
            # transforms.Lambda(random_rotate),
            ToTensor()
        ])
    ts_transform = transforms.Compose([
        ToTensor()
    ])

    tr_loader = get_data_loader(args, data_split='train', transform=tr_transform, subset=args.subset)
    vl_loader = get_data_loader(args, data_split='val', transform=ts_transform, subset=args.subset)
    # vl_loader = get_data_loader(args, data_split='test',   transform=ts_transform, subset=args.subset)  # RPV
    ts_loader = get_data_loader(args, data_split='test', transform=ts_transform, subset=args.subset)

    args.log_file.write("Number of training samples: %d\n" % len(tr_loader.dataset))
    args.log_file.write("Number of validation samples: %d\n" % len(vl_loader.dataset))
    args.log_file.write("Number of testing samples: %d\n" % len(ts_loader.dataset))

    args.log_file.write("--------------------------------------------------\n")
    args.log_file.close()

    # if args.evaluate:
    #     # 将两个 criterion 传入 validate 函数
    #     validate(ts_loader, model, criterion_factual, args, valid_set="Test")
    #     return
    #
    # if args.fp16:
    #     args.scaler = torch.cuda.amp.GradScaler()
    #
    # cont_epoch = 0
    # best_epoch = 0
    # test_acc2 = 0
    #
    # for epoch in range(args.start_epoch, args.epochs):
    #
    #     args.log_file = open(os.path.join(args.ckpt, "log.txt"), mode="a")
    #
    #     # train for one epoch
    #     # 将两个 criterion 传入 train 函数
    #     train(tr_loader, model, criterion_factual, criterion_cf, optimizer, epoch, args)
    #
    #     # evaluate on validation set
    #     acc1 = validate(vl_loader, model, criterion_factual, args, valid_set="Valid")
    #
    #     # remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)
    #
    #     if is_best:
    #         acc2 = validate(ts_loader, model, criterion_factual, args, valid_set="Test")
    if args.evaluate:
        # sim
        # evaluate_cosine_similarity(ts_loader,model,args)
        acc = validate(ts_loader, model, criterion, args, valid_set="Test")
        return

    if args.fp16:
        args.scaler = torch.cuda.amp.GradScaler()

    cont_epoch = 0
    best_epoch = 0
    test_acc2  = 0


    for epoch in range(args.start_epoch, args.epochs):

        args.log_file = open(log_path, mode="a")

        # train for one epoch
        train(tr_loader, model, criterion, optimizer, epoch, args, csv_log_path)

        # evaluate on validation set
        acc1 = validate(vl_loader, model, criterion, args, csv_log_path, epoch, len(tr_loader), valid_set="Valid")

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            acc2 = validate(ts_loader, model, criterion, args, csv_log_path, epoch, len(tr_loader), valid_set="Test")

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc": best_acc1,
            "optimizer": optimizer.state_dict(),
        }, is_best, epoch, save_path=args.ckpt)

        if is_best:
            cont_epoch = 0
            best_epoch = epoch
            test_acc2 = acc2
        else:
            cont_epoch += 1

        epoch_msg = ("----------- Best Acc at [{}]: Valid {:.3f} Test {:.3f} Continuous Epoch {} -----------".format(
            best_epoch, best_acc1, test_acc2, cont_epoch)
        )
        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")
        args.log_file.close()


def train(data_loader, model, criterion, optimizer, epoch, args, csv_log_path):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1],
        prefix = "Epoch: [{}]".format(epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global_step = epoch * len(data_loader) + i

        optimizer.zero_grad()

        if args.gpu is not None:
            images = images.to(args.device, non_blocking=True)
        if torch.cuda.is_available():
            target = target.to(args.device, non_blocking=True)

        images = normalize_image(images)

        # # compute output
        # if args.fp16:
        #     with torch.amp.autocast('cuda'):
        #         # ret = model(images)
        #         #
        #         # # 兼容: 模型可能返回 Tensor 或 (Tensor, aux)
        #         # if isinstance(ret, tuple):
        #         #     logits, aux = ret
        #         # else:
        #         #     logits, aux = ret, {}
        #         #
        #         # # 主损失 -> 标量
        #         # main_loss = criterion(logits, target)
        #         # if torch.is_tensor(main_loss) and main_loss.dim() > 0:
        #         #     main_loss = main_loss.mean()
        #         #
        #         # # 解析辅助损失（可能来自不同副本，先聚合为标量）
        #         # dag_loss = 0.0
        #         # tc_loss = 0.0
        #         #
        #         # if isinstance(aux, dict):
        #         #     v = aux.get("dag_loss", None)
        #         #     if v is not None:
        #         #         dag_loss = v.mean() if torch.is_tensor(v) and v.dim() > 0 else v
        #         #     v = aux.get("tc_loss", None)
        #         #     if v is not None:
        #         #         tc_loss = v.mean() if torch.is_tensor(v) and v.dim() > 0 else v
        #         # elif torch.is_tensor(aux):
        #         #     # 若 DataParallel 把每卡的 dag_loss 堆成 [num_replicas]，这里做平均
        #         #     dag_loss = aux.mean()
        #         #
        #         # # 转成 tensor 并放到正确设备/类型
        #         # if not torch.is_tensor(dag_loss):
        #         #     dag_loss = torch.tensor(float(dag_loss), device=logits.device, dtype=main_loss.dtype)
        #         # if not torch.is_tensor(tc_loss):
        #         #     tc_loss = torch.tensor(float(tc_loss), device=logits.device, dtype=main_loss.dtype)
        #         #
        #         # lambda_tc = getattr(args, "lambda_tc", 1e-3)
        #         # lambda_dag = getattr(args, "lambda_dag", 1e-2)
        #         #
        #         # loss = main_loss + lambda_tc * tc_loss + lambda_dag * dag_loss
        #         # # 最终兜底，确保是标量
        #         # if loss.dim() > 0:
        #         #     loss = loss.mean()
        #
        #         # out = model(images)
        #         #
        #         # # 兼容 logits / (logits, aux)
        #         # if isinstance(out, (tuple, list)):
        #         #     logits = out[0]
        #         #     aux = out[1] if len(out) > 1 else {}
        #         # else:
        #         #     logits, aux = out, {}
        #         #
        #         # loss = criterion(logits, target)
        #         # if torch.is_tensor(loss) and loss.dim() > 0:
        #         #     loss = loss.mean()  # ★ 保证是标量
        #         #
        #         # # 处理 tc_loss（DataParallel 会把多卡标量堆成向量，这里做均值）
        #         # if isinstance(aux, dict) and ('tc_loss' in aux) and (aux['tc_loss'] is not None):
        #         #     tc = aux['tc_loss']
        #         #     if torch.is_tensor(tc) and tc.dim() > 0:
        #         #         tc = tc.mean()
        #         #     loss = loss + args.tc_weight * tc  # ★ 仍是标量
        #     args.scaler.scale(loss).backward()
        #     args.scaler.step(optimizer)
        #     args.scaler.update()
        # else:
        #     out = model(images)
        #     logits, aux = (out if isinstance(out, (tuple, list)) else (out, {}))
        #     loss = criterion(logits, target)
        #     if isinstance(aux, dict) and ('tc_loss' in aux):
        #         loss = loss + args.tc_weight * aux['tc_loss']
        #     loss.backward()
        #     optimizer.step()
        # # measure accuracy
        # acc1 = accuracy(logits, target)

        # compute output
        if args.fp16:
            with torch.cuda.amp.autocast():

                output, all_errors = model(images)
                # output = model(images)
                loss = criterion(output, target)

            args.scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            args.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            args.scaler.step(optimizer)
            args.scaler.update()
        else:
            output = model(images)
            loss = criterion(output, target)
            # compute gradient and do SGD step
            loss.backward()

            args.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # log_errors_to_csv(csv_log_path, epoch, global_step, 'train', all_errors)
        with torch.no_grad():  # 在不计算梯度的模式下进行，以节省资源
            b = images.size(0)  # 获取 batch size

            # 1. 找出答对和答错的样本
            _, pred = output.topk(1, 1, True, True)
            correct_mask = pred.view(-1).eq(target)  # 布尔掩码, shape: [batch_size]

            # 2. 遍历每一层的误差张量
            for layer_idx, error_tensor in enumerate(all_errors):
                # error_tensor 的 shape 类似于 [batch_size * 8, channels, ...]
                # 我们需要将 correct_mask 扩展以匹配 error_tensor 的维度
                # 每个样本对应8个选项的误差，所以将掩码的每个元素重复8次
                num_choices = error_tensor.shape[0] // b
                expanded_mask = correct_mask.repeat_interleave(num_choices)

                # 3. 根据掩码分离误差
                correct_errors = error_tensor[expanded_mask]
                incorrect_errors = error_tensor[~expanded_mask]

                # 4. 分别计算并记录
                if correct_errors.numel() > 0:
                    mean_err = torch.mean(torch.abs(correct_errors)).cpu().item()
                    log_errors_to_csv(csv_log_path, epoch, global_step, 'train', layer_idx, 'correct', mean_err)

                if incorrect_errors.numel() > 0:
                    mean_err = torch.mean(torch.abs(incorrect_errors)).cpu().item()
                    log_errors_to_csv(csv_log_path, epoch, global_step, 'train', layer_idx, 'incorrect', mean_err)


        # measure accuracy and record loss
        acc1 = accuracy(output, target)

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0][0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            epoch_msg = progress.get_message(i+1)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

            args.log_file.write(epoch_msg + "\n")


def validate(data_loader, model, criterion, args, csv_log_path, epoch, train_loader_len, valid_set='Valid', ):
    if 'RAVEN' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    elif 'Unicode' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    elif 'MNR' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    else:
        acc_regime = None

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1],
        prefix=valid_set + ': ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):

            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(args.device, non_blocking=True)

            images = normalize_image(images)

            # # compute outputs
            # output = model(images)
            # output = model(images, train=False)
            # compute outputs
            output, all_errors = model(images)  # 获取 all_errors

            # # 仅在验证集上记录 (避免重复记录测试集)
            # if valid_set == 'Valid':
            #     global_step = (epoch + 1) * train_loader_len
            #     log_errors_to_csv(csv_log_path, epoch, global_step, 'valid', all_errors)
            if valid_set == 'Valid':
                b = images.size(0)
                global_step = (epoch + 1) * train_loader_len

                _, pred = output.topk(1, 1, True, True)
                correct_mask = pred.view(-1).eq(target.view(-1))

                for layer_idx, error_tensor in enumerate(all_errors):
                    num_choices = error_tensor.shape[0] // b
                    expanded_mask = correct_mask.repeat_interleave(num_choices)

                    correct_errors = error_tensor[expanded_mask]
                    incorrect_errors = error_tensor[~expanded_mask]

                    if correct_errors.numel() > 0:
                        mean_err = torch.mean(torch.abs(correct_errors)).cpu().item()
                        log_errors_to_csv(csv_log_path, epoch, global_step, 'valid', layer_idx, 'correct', mean_err)

                    if incorrect_errors.numel() > 0:
                        mean_err = torch.mean(torch.abs(incorrect_errors)).cpu().item()
                        log_errors_to_csv(csv_log_path, epoch, global_step, 'valid', layer_idx, 'incorrect', mean_err)


            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(output, target)

            top1.update(acc1[0][0], images.size(0))

            if acc_regime is not None:
                update_acc_regime(args.dataset_name, acc_regime, output, target, structure_encoded, data_file)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(data_loader) - 1:
                epoch_msg = progress.get_message(i + 1)
                print(epoch_msg)

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

            mean_acc = 0
            valid_regimes = 0
            for key, val in acc_regime.items():
                if val is not None:
                    mean_acc += val
                    valid_regimes += 1
                # mean_acc += val
            if valid_regimes > 0:
                mean_acc /= valid_regimes
            else:
                mean_acc /= top1.avg
            # mean_acc /= len(acc_regime.keys())
        else:
            mean_acc = top1.avg

        epoch_msg = '----------- {valid_set} Acc {mean_acc:.3f} -----------'.format(
            valid_set=valid_set, mean_acc=mean_acc
        )

        print(epoch_msg)

        if args.evaluate == False:
            args.log_file.write(epoch_msg + "\n")

    if args.show_detail:
        for key, val in acc_regime.items():
            print("configuration [{}] Acc {:.3f}".format(key, val))
    return mean_acc


if __name__ == '__main__':
    main()