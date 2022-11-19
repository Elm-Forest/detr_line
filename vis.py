import matplotlib
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from skimage import io
matplotlib.use('TkAgg')

train_json = 'D:/dataset/coco_powerline_1/annotations/train.json'
train_path = 'D:/dataset/coco_powerline_1/train/'


def visualization_seg(num_image, json_path):
    # 需要画图的是第num副图片, 对应的json路径和图片路径,
    # str = ' '为类别字符串，输入必须为字符串形式 'str'，若为空，则返回所有类别id
    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=['str'])  # 获取指定类别 id
    imgIds = coco.getImgIds(catIds=catIds)  # 获取图片i
    img = coco.loadImgs(imgIds[num_image - 1])[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    image = io.imread(train_path + img['file_name'])

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # 读取在线图片的方法
    # I = io.imread(img['coco_url'])

    plt.imshow(image)
    coco.showAnns(anns)
    plt.show()


if __name__ == "__main__":
    visualization_seg(136, train_json)
