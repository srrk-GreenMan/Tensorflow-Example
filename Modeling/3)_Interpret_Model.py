import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential, Input

"""
Reference : https://keras.io/examples/vision/grad_cam/
저번 시간에는 pretrain 모델을 원하는 만큼 불러오는 방법들에 대해서 배웠습니다. 
이번 시간에는 pretrain 모델을 사용해서 grad-cam을 적용하는 방법들에 대해서 배워보겠습니다. 

우선 저번 시간에 사용한 모바일넷2를 들고 오겠습니다. 
"""

model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=True
)

model.summary()

last_conv_layer_name = "Conv_1"

# 귀여운 코끼리 사진을 가져오겠습니다.
img_path = tf.keras.utils.get_file(
    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
)

# Grad CAM Algorithm

@tf.function
def read_image(img_path, size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.expand_dims(img, axis=0)
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.Model(
        model.input, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if __name__ == "__main__":
    img_array = read_image(img_path, (224, 224))
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    plt.imsave('HeatMap.jpg', heatmap)