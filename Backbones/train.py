import os
import sys
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from Models.ResNet import resnet34, resnet50, resnet101
from Models.VGGNet import vgg11, vgg13, vgg16, vgg19


available_models = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16':  vgg16,
    'vgg19':  vgg19,
}

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomRotation(10),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    datasets_path = os.path.join(os.getcwd(), 'Datasets/cifar10')
    assert os.path.exists(datasets_path), "{} path does not exist.".format(datasets_path)

    # Download CIFAR10
    train_dataset = datasets.CIFAR10(root=datasets_path,
                                     train=True,
                                     download=False,
                                     transform=data_transform['train'])
    validate_dataset = datasets.CIFAR10(root=datasets_path,
                                    train=False,
                                    download=False,
                                    transform=data_transform['val'])

    # Get classes index
    class_list = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in class_list.items())
    # Write dict into json file
    json_str = json.dumps(class_dict, indent=4) # indent=4, 在每一行前面加上4个空格
    with open('cifar10_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    choose_model = args.choose_model
    assert choose_model in available_models, "Error: Model {} not in available_models".format(choose_model)

    # Choose the corresponding model
    net = available_models.get(choose_model)()
    # Parallel training
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Using {} GPUs".format(torch.cuda.device_count()))
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    epochs = args.epochs
    best_acc = 0.0
    save_path = './checkpoints/{}.pth'.format(choose_model)

    train_steps = len(train_loader)
    loss_list = []
    iter_list = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            iter_list.append(epoch*len(train_bar) + step)
            loss_list.append(loss.item())

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(validate_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    # plt.plot(loss_list, "Loss")
    # plt.plot(iter_list, "iter")
    # plt.title('Loss')
    # plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int,help='number_of_epochs')
    # batch size
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    # learning rate 第6轮开始把学习率往下调0.33倍，然后每3轮调一次，一共20轮
    parser.add_argument('--lr', default=0.001, type=float, help='learing_rate')
    # choose model
    parser.add_argument('--choose_model', default='resnet34', help='choose_model')

    args = parser.parse_args()
    print(args)

    main(args)
