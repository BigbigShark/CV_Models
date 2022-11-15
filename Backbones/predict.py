import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob

from Models.ResNet import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Read class_indict
    json_path = 'cifar10_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Load image
    img_file_path = os.path.join(os.getcwd(), 'test_imgs')
    assert os.path.exists(img_file_path), "file: '{}' does not exist.".format(img_file_path)
    img_path = glob('{}/*.*'.format(img_file_path))

    # Create model
    model = resnet34(10).to(device)

    # Load model weights
    weights_path = os.path.join(os.getcwd(), 'checkpoints', 'resnet34_0.pth')
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # Prediction
    model.eval()

    for single_img_path in img_path:
        if single_img_path.endswith('jpg') or single_img_path.endswith('JPEG'):
            img = Image.open(single_img_path)
            plt.show()
            # [N, C, H, W]
            img = data_transform(img)
            # Expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)

            with torch.no_grad():
                # Predict class
                output = torch.squeeze(model(img)).cpu()
                predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
            plt.title(print_res)
            for i in range(len(predict)):
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
            plt.show()


if __name__ == '__main__':
    main()
