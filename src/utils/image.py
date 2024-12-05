import torch
import numpy as np
import matplotlib.pyplot as plt


# 图像转化
def _im_convert(tensor: torch.Tensor):

    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# 画图
def show_result(images:np.ndarray, preds:np.ndarray, labels:np.ndarray, rows:int = 2, columns:int = 4, figsize:tuple[float,float] = (20,20)):
    fig = plt.figure(figsize=figsize)
    cat_to_name = {
        '0':'奶龙',
        '1':'并非奶龙'
    }
    for index in range(columns * rows):
        ax = fig.add_subplot(rows, columns, index + 1, xticks=[], yticks=[])
        plt.imshow(_im_convert(images[index]))
        pred = cat_to_name[str(preds[index])]
        actual = cat_to_name[str(labels[index].item())]
        color = 'green' if pred == actual else "red"
        title = "{} ({})".format(pred,actual)
        ax.set_title(title, color=color)

    plt.show()