```shell
python main.py
 --coco_path D:\dataset\coco2017
 --coco_panoptic_path D:\dataset\coco_panoptic
 --masks
 --dataset_file coco_panoptic 
 --output_dir ./output
```

### Powerline Detection

```shell
python main.py
 --epochs 500
 --dataset_file coco_powerline
 --batch_size 2
 --num_classes 2
 --num_queries 10
 --coco_path D:\dataset\coco_powerline_1
 --output_dir ./output
 --resume ./output/checkpoint.pth
```

### Powerline Seg

```shell
python main.py
 --masks
 --epochs 500
 --dataset_file coco_powerline
 --batch_size 2
 --num_classes 4
 --num_queries 10
 --coco_path D:\dataset\coco_powerline_1
 --output_dir ./output/seg
```
