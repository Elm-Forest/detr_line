import json

file_name = 'train.json'
# 读取COCO数据集标注文件
with open(f'D:/dataset/coco_powerline_1/annotations/ori/{file_name}', 'r') as f:
    annotations = json.load(f)

i = 0
# 遍历标注列表，删除电塔类别的标注
filtered_annotations = []
for annotation in annotations['annotations']:
    # # 电线类别的category_id为0
    if annotation['category_id'] != 1 and annotation['category_id'] != 2 and annotation['category_id'] != 3:
        filtered_annotations.append(annotation)



for annotation in filtered_annotations:
    annotation['category_id'] = 1


# print(i)
# for k in filtered_annotations:
#     print(k)

# 更新标注列表
annotations['annotations'] = filtered_annotations

# 保存删除电塔类别标注后的新标注文件
with open(f'D:/dataset/coco_powerline_1/annotations/new/{file_name}', 'w') as f:
    json.dump(annotations, f)
