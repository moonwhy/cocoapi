import json
import os
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/home/ubuntu/user_space/cocoapi/PythonAPI/pycocotools/mydata/data/'
# 用于创建训练集或验证集
phase = 'train'
# 训练集和验证集划分的界线
split = 130
#
dataset = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances',
             "info": {"description": "This is fod data with the 2014 MS COCO dataset style.",
                      "version": "1.0", "year": 2019, "date_created": "2019-11-6 22:20"},
           }

# 打开类别标签
with open(os.path.join(root_path, 'classes.names')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
_indexes = [f for f in os.listdir(os.path.join(root_path, 'images'))]

# 判断是建立训练集还是验证集 ,获得相关图像和标记文件名
if phase == 'train':
    img_files = [line for i, line in enumerate(_indexes) if i <= split]
    label_files = [
        path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        for path in img_files
    ]
elif phase == 'val':
    img_files = [line for i, line in enumerate(_indexes) if i > split]
    label_files = [
        path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        for path in img_files
    ]
else:
    img_files = _indexes
    label_files = [
        path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        for path in img_files
    ]

width = 2560
height = 1440
# 读取Bbox信息
for k, index in enumerate(img_files):
    # 用opencv读取图片，得到图像的宽和高
    img = cv2.imread(os.path.join(root_path, 'images/') + index)
    height, width, _ = img.shape

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})

    with open(os.path.join(root_path, 'labels/', label_files[k])) as tr:
        annos = tr.readlines()
    if annos == '':
        dataset['annotations'].append({
            'area': 0,
            'bbox': [],
            'category_id': 0,
            'id': 1,
            'image_id': k,
            'iscrowd': 0,
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': []
        })
    else:
        for anno in annos:
            parts = list(anno.strip().split())  # [class_ind, center_i, center_j, box_w, box_h]

            # 类别
            cls_id = int(parts[0])
            # x_min
            x1 = width * (float(parts[1]) - float(parts[3]) / 2)
            # y_min
            y1 = height * (float(parts[2]) - float(parts[4]) / 2)
            # x_max
            x2 = width * (float(parts[1]) + float(parts[3]) / 2)
            # y_max
            y2 = height * (float(parts[2]) + float(parts[4]) / 2)
            box_width = x2 - x1
            box_height = y2 - y1
            dataset['annotations'].append({
                'area': box_width * box_height,
                'bbox': [x1, y1, box_width, box_height],
                'category_id': int(cls_id),
                'id': 1,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })

            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = classes[cls_id]
            unique_color = [255, 255, 255]

            # add the bbox to the img
            cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, 2)

            text_label = '{}'.format(class_name)
            # text_label = '{}'.format(classes[int(cls_pred)])
            (ret_val, base_line) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)  # cv2.FONT_HERSHEY_COMPLEX
            text_org = (x1, y1 - 0)
            # draw text rect
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + 5),  # + base_line
                          (text_org[0] + int(ret_val[0] * 0.6) + 5, text_org[1] - int(ret_val[1] * 0.6) - 5), unique_color,
                          2)
            # this rectangle for fill text rect
            cv2.rectangle(img, (text_org[0] - 5, text_org[1] + 5),  # + base_line
                          (text_org[0] + int(ret_val[0] * 0.6) + 5, text_org[1] - int(ret_val[1] * 0.6) - 5),
                          unique_color, -1)
            cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)
            cv2.imshow('result.jpg',img)

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)

'''
————————————————
版权声明：本文为CSDN博主「Jayce~」的原创文章，遵循CC4.0BY - SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / qq_15969343 / article / details / 80848175
'''