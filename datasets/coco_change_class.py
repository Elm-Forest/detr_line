import json

file_name = 'test.json'

# 读取COCO数据集标注文件
with open(f'D:/dataset/coco_powerline_1/annotations/new/{file_name}', 'r') as f:
    annotations = json.load(f)

# 遍历标注列表，将category_id为0的物体改为1
for annotation in annotations['annotations']:
    if annotation['category_id'] == 0:
        annotation['category_id'] = 1

# 保存修改后的标注文件
with open(f'D:/dataset/coco_powerline_1/annotations/new/{file_name}', 'w') as f:
    json.dump(annotations, f)
