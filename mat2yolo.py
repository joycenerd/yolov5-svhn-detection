import h5py
import numpy as np
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/vrdl_hw2_data', help='data root dir')
args = parser.parse_args()


def get_img_name(f, name_col, idx=0):
    img_name = ''.join(map(chr, f[name_col[idx][0]][()].flatten()))
    return (img_name)


def get_img_boxes(f, bbox_col, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    bbox_prop = ['height', 'left', 'top', 'width', 'label']
    meta = {key: [] for key in bbox_prop}

    box = f[bbox_col[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta


def yolo_annot(bbox, img_path, annot_path):
    annot_f = open(annot_path, 'w')

    img = Image.open(img_path)
    w, h = img.size

    label_cnt = len(bbox['label'])
    for i in range(label_cnt):
        label, height, left, top, width = bbox['label'][i], bbox['height'][i], bbox['left'][i], bbox['top'][i], \
                                          bbox['width'][i]
        if label == 10:
            label = 0

        x_center = left + width / 2
        y_center = top + height / 2
        x_center_norm = x_center / w
        y_center_norm = y_center / h

        width_norm = width / w
        height_norm = height / h

        annot_f.write(f'{label} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n')
    annot_f.close()


if __name__ == '__main__':
    annot_dir = os.path.join(args.data_root, 'labels/all_train')
    if not os.path.isdir(annot_dir):
        os.makedirs(annot_dir)

    mat_f = os.path.join(args.data_root, 'train/digitStruct.mat')
    mat = h5py.File(mat_f)

    data_size = mat['/digitStruct/name'].shape[0]
    print(f'Data size: {data_size}')

    name_col = mat['/digitStruct/name']
    bbox_col = mat['/digitStruct/bbox']

    for idx in range(data_size):
        img_name = get_img_name(mat, name_col, idx)
        bbox = get_img_boxes(mat, bbox_col, idx)
        print(img_name, bbox)
        annot_f = os.path.join(annot_dir, f'{img_name[:-3]}txt')
        img_path = os.path.join(args.data_root, 'train', img_name)
        yolo_annot(bbox, img_path, annot_f)
