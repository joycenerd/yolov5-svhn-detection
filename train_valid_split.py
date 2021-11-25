import glob
import argparse
import math
import random
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/vrdl_hw2_data',
                    help='trainig image saving directory')
parser.add_argument('--ratio', type=float, default=0.2, help='validation data ratio')
args = parser.parse_args()

if __name__ == '__main__':
    # if not os.path.isdir(args.data_dir):
    #     os.makedirs(args.data_dir)

    src_img_dir = os.path.join(args.data_root, 'train')
    data_size = len(glob.glob1(src_img_dir, "*.png"))
    valid_size = math.floor(data_size * args.ratio)

    img_list = []
    for img_path in glob.glob(f'{src_img_dir}/*.png'):
        img_list.append(img_path)

    idx = random.sample(range(data_size), valid_size)

    # train_rec_path=os.path.join(args.data_dir,'train.txt')
    # valid_rec_path=os.path.join(args.data_dir,'validation.txt')
    # train_rec=open(train_rec_path,'w')
    # valid_rec=open(valid_rec_path,'w')

    dest_img_dir = os.path.join(args.data_root, 'images')
    train_img_dir = os.path.join(dest_img_dir, 'train')
    valid_img_dir = os.path.join(dest_img_dir, 'valid')
    src_label_dir = os.path.join(args.data_root, 'labels/all_train')
    train_label_dir = src_label_dir.replace('all_train', 'train')
    valid_label_dir = src_label_dir.replace('all_train', 'valid')
    if not os.path.isdir(dest_img_dir):
        os.mkdir(dest_img_dir)
        os.mkdir(train_img_dir)
        os.mkdir(valid_img_dir)
        os.mkdir(train_label_dir)
        os.mkdir(valid_label_dir)

    for i in range(data_size):
        if i in idx:
            src_img = img_list[i]
            dest_img = src_img.replace('train', 'images/valid')
            shutil.copy(src_img, dest_img)
            src_label = src_img.replace('train', 'labels/all_train').replace('png', 'txt')
            dest_label = src_label.replace('all_train', 'valid')
            shutil.copyfile(src_label, dest_label)
        else:
            src_img = img_list[i]
            dest_img = src_img.replace('train', 'images/train')
            shutil.copy(src_img, dest_img)
            src_label = src_img.replace('train', 'labels/all_train').replace('png', 'txt')
            dest_label = src_label.replace('all_train', 'train')
            shutil.copyfile(src_label, dest_label)

    train_size = len(glob.glob1(train_img_dir, "*.png"))
    valid_size = len(glob.glob1(valid_img_dir, "*.png"))
    print(f'train size: {train_size}\tvalid size: {valid_size}')
