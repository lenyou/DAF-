## second_homework
# Method
采用图像处理的方案，类似于人脸识别的triple loss， 首先需要将文章信息转化为图片，具体事例在example里面。然后在训练时， 这个模型会同时优化俩个loss，一个是sameloss，负责将同一作者的论文的距离为0. 另一个loss是marginloss，就是讲不同的作者的距离要比同一作者的距离总是大于2.

