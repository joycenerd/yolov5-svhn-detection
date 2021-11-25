import argparse
import os
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument('--yolo-path', type=str, default='yolov5/runs/detect/exp2/labels')
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/vrdl_hw2_data')
args = parser.parse_args()

if __name__ == '__main__':
    img_dir = os.path.join(args.data_root, 'test')
    data_listdir = os.listdir(img_dir)
    data_listdir.sort(key=lambda x: int(x[:-4]))

    result_to_json = []
    for img_name in data_listdir:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        image_id = int(img_name[:-4])
        label_path = os.path.join(args.yolo_path, img_name.replace('png', 'txt'))
        if not os.path.exists(str(label_path)):
            continue
        label = open(label_path, 'r')
        lines = label.readlines()

        for line in lines:
            det_box_info = {}

            det_box_info['image_id'] = image_id
            line = line.strip().split()
            x_center, y_center, width, height = float(line[1]), float(line[2]), float(line[3]), float(line[4])

            x_center *= w
            y_center *= h
            width *= w
            height *= h
            min_x = x_center - width / 2
            min_y = y_center - height / 2

            det_box_info['bbox'] = (tuple((min_x, min_y, width, height)))
            det_box_info['score'] = float(line[5])
            det_box_info['category_id'] = int(line[0])
            result_to_json.append(det_box_info)

        label.close()

    # Write the list to answer.json 
    json_object = json.dumps(result_to_json, indent=4)

    with open("answer.json", "w") as outfile:
        outfile.write(json_object)
