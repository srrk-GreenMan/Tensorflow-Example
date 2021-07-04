import tensorflow as tf


"""
이번 시간에는 간단한 커스텀 레이어를 만들어 보겠습니다. 해당 내용은 Upsampling2D와 비슷합니다. 

ToBe
[1, 2, 3, 4] -> 각 채널별로 해당 값들이 가득차있는 3차원 텐서 (width, height, 4)
"""

class Custom_Layer(tf.keras.layers.Layer):
    def __init__(self, w, h):
        super(Custom_Layer, self).__init__()
        self.width = w
        self.height = h

    def call(self, input):
        a, b, c, d = map(lambda x: tf.fill((self.width, self.height), x), input)
        result = tf.stack([a, b, c, d])
        # Channel-Last로 변경
        result = tf.transpose(result, perm=[1, 2, 0])
        return result

layer = Custom_Layer(10, 10)
a = layer([1, 2, 3, 4])
print(a[:, :, 0])
"""
<tf.Tensor: shape=(10, 10), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)>
"""
