import os
import datetime
# 0->1, 1->2, 2->3, 3->0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch import nn

from network import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from train_utils import newDataset
from train_utils import transforms
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_val_utils as utils
from train_utils import plot_map, plot_loss_and_lr


def create_model(num_classes, load_pretrain_weights=True):
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(pretrain_path="./backbone/resnet50.pth", trainable_layers=5)

    model = FasterRCNN(backbone=backbone, num_classes=91) # 该91不要修改

    if load_pretrain_weights:
        # 载入预训练模型权重
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # if torch.cuda.device_count() > 1:
    #     print("using {} cudas".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    if not os.path.exists("./info_of_train"):
        os.mkdir("./info_of_train")
    results_file = "./info_of_train/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    datasets_root = './datasets'
    assert os.path.exists(datasets_root), "file {} does not exist.".format(datasets_root)

    # Load train datasets
    train_dataset = newDataset(datasets_root=datasets_root, transforms=data_transform["train"], txt_name="train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        collate_fn=train_dataset.collate_fn)

    # Load validation datasets
    val_dataset = newDataset(datasets_root=datasets_root, transforms=data_transform["val"], txt_name="val.txt")
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      collate_fn=val_dataset.collate_fn)

    # Create model, num_classes = num_classes + background
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=True)

    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad] # 需要训练的参数才送进优化器中
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Scheduler of learning rate: 每step_size个epochs后，将lr更新为原来的gamma倍
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_loss = []
    learning_rate = []
    val_map = []
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for epoch in range(args.epochs):
        # Train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=10, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        coco_info = utils.evaluate(model, val_dataset_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if epoch > 9:
            torch.save(save_files, "{}/ResNetFpn-Model-Epoch{}.pth".format(args.output_dir, epoch))

    if not os.path.exists("./train_result"):
        os.mkdir("./train_result")
    # Plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--datasets_dir', default='./datasets', help='root_path_of_datasets')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=10, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output_dir', default='./save_weights', help='path_to_save_weights')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number_of_epochs')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='batch_size')
    # aspect-ratio-group-factor
    parser.add_argument('--aspect_ratio_group_factor', default=3, type=int, help='aspect_ratio_group_factor')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)