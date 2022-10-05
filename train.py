import datetime
import os
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
import torchvision.datasets.video_utils
from torchvision.datasets.samplers import DistributedSampler, UniformClipSampler, RandomClipSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from tqdm import tqdm
import presets
import utils

from scheduler import WarmupMultiStepLR

try:
    from apex import amp
except ImportError:
    amp = None


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    n_batches = len(data_loader)

    with tqdm(data_loader, unit="batch") as tepoch:
        total_acc1, total_acc5, total_loss = 0, 0, 0

        for video, target in tepoch:
            video, target = video.to(device), target.to(device)
            output = model(video)
            loss = criterion(output, target)

            optimizer.zero_grad()
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            total_loss += loss.item()

            tepoch.set_postfix(loss=loss.item(), accuracy1=acc1.item(), accuracy5=acc5.item())
            lr_scheduler.step()

    return total_acc1 / n_batches, total_acc5 / n_batches, total_loss / n_batches

def evaluate(model, criterion, data_loader, device):
    model.eval()

    total_acc1, total_acc5, total_loss = 0, 0, 0
    n_batches = len(data_loader)

    with torch.no_grad():
        for video, target in data_loader:
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()
            total_loss += loss.item()

    return total_acc1 / n_batches, total_acc5 / n_batches, total_loss / n_batches


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    traindir = os.path.join(args.data_path, args.train_dir)
    valdir = os.path.join(args.data_path, args.val_dir)

    print("Loading training data")
    cache_path = _get_cache_path(traindir)
    transform_train = presets.VideoClassificationPresetTrain((128, 171), (112, 112))

    if args.cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        dataset.transform = transform_train
    else:
        if args.distributed:
            print("It is recommended to pre-compute the dataset cache "
                  "on a single-gpu first, as it will be faster")

        dataset = torchvision.datasets.UCF101(
            '/content/action_recognition_R2plus1d/dataset/UCF-101',
            '/content/action_recognition_R2plus1d/dataset/ucfTrainTestlist',
            frames_per_clip=16, step_between_clips=5
        )
        if args.cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)

    transform_test = presets.VideoClassificationPresetEval((128, 171), (112, 112))

    if args.cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
        dataset_test.transform = transform_test
    else:
        if args.distributed:
            print("It is recommended to pre-compute the dataset cache "
                  "on a single-gpu first, as it will be faster")

        dataset_test = torchvision.datasets.UCF101(
            '/content/action_recognition_R2plus1d/dataset/UCF-101',
             '/content/action_recognition_R2plus1d/dataset/ucfTrainTestlist',
            frames_per_clip=16, step_between_clips=5, train=False
        )
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
    test_sampler = UniformClipSampler(dataset_test.video_clips, args.clips_per_video)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)
        test_sampler = DistributedSampler(test_sampler)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn)

    print("Creating model")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr * args.world_size
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_acc1, train_acc5, train_loss = train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq, args.apex)
        eval_acc1, eval_acc5, eval_loss = evaluate(model, criterion, data_loader_test, device=device)

        print(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4}, Train Accuracy@1: {train_acc1:.4}"\
            + f", Train Accuracy@5: {train_acc5:.4} Val Loss: {eval_loss: .4}, Val Accuracy@1: {eval_acc1:.4}"\
            + f", Val Accuracy@5: {eval_acc5:.4}"
        )


        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Video Classification Training')

    parser.add_argument('--data-path', default='/datasets01_101/kinetics/070618/', help='dataset')
    parser.add_argument('--train-dir', default='train_avi-480p', help='name of train dir')
    parser.add_argument('--val-dir', default='val_avi-480p', help='name of val dir')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=24, type=int)
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
