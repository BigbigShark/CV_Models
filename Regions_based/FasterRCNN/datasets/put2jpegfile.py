'''将tiny_vid数据集中的所有照片放到datasets/JPEGImgs目录下'''
import os
import shutil

all_classes = ['bird', 'car', 'dog', 'lizard', 'turtle']

def main():
    for class_name in all_classes:
        img_file = r'.\tiny_vid\{}'.format(class_name)
        i = 0
        for img_name in os.listdir(img_file):
            # img_num = img_name.split(".")[0]
            i = i + 1
            if i > 180:
                break
            new_img_name = "{}{}".format(class_name, i)
            src_path = img_file + '\\' + img_name
            dst_path = r'.\JPEGImages\{}.JPEG'.format(new_img_name)
            shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    main()