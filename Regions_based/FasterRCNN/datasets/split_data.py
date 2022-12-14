import os
import random

def main():
    random.seed(2022)  # 设置随机种子，保证随机结果可复现

    file_path = "./Annotations"
    assert os.path.exists(file_path), "path: '{}' does not exist.".format(file_path)

    val_rate = 0.2

    files_name = sorted([file.split(".")[0] for file in os.listdir(file_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        train_files.append(file_name)
    random.shuffle(train_files)
    try:
        train_f = open("./train.txt", "x")
        eval_f = open("./val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)

if __name__ == '__main__':
    main()