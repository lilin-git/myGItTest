
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


from gender.vgg11_model_age import vgg16_model_fn


PATH = "data/predict/"

def data(path):
    floder_names = os.listdir(path)
    img = Image.open(path+floder_names[0])
    # 默认读取是 gray，需要转换
    img = img.convert("RGB")
    # 把图片转换为合适大小
    img = img.resize((128, 128))
    _data = img.getdata()
    _data = np.array(_data, dtype=np.float32)
    images = np.reshape(_data, (-1, 49152))
    return images

def data_vi(img):
    img = cv2.resize(img, (128, 128))
    img = np.array(img, dtype=np.float32)
    # img = img.resize((224, 224))
    images = np.reshape(img, (-1, 49152))
    return images

def age_model(model_path, img=None):
    """
    测试model准确率
    """

    # eval_data = data(PATH)
    eval_data = data_vi(img)
    # 创建模型
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=vgg16_model_fn, model_dir=model_path)

    # 评估模型和输出结果
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        num_epochs=1,
        shuffle=True
    )
    eval_results = cifar10_classifier.predict(input_fn=eval_input_fn)
    eval_results = list(eval_results)
    # print(eval_results)
    age_index = int(eval_results[0]['classes'])
    agess = [6,14,18,20,22,24,30,42,66,78]
    return agess[age_index]


if __name__ == "__main__":
    result = age_model("models/vgg_model")
    print(result)