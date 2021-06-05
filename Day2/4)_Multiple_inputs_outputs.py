import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input

"""
모델을 만들면 하나의 input, output을 선정하지 않는 경우가 있습니다. 이번 시간에는 그러한 문제들을 해결하는 방법에 대해서 배워보겠습니다.
"""

# 간단한 함수 식을 추정해보겠습니다.
# y = 3 * x1 + 4 * y1 이라는 식을 다음과 같은 데이터로 추정해보겠습니다.
# 텐서플로우에서 multiple inputs를 넣어주기 위해선 딕셔너리 형태로 데이터를 만드는 것이 편합니다.
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x_set = [(i, j) for i in x1 for j in x2]

y = np.array([3*i + 4*j for i in x1 for j in x2]).reshape(-1, 1)
_x1 = np.array([i for i, j in x_set]).reshape(-1, 1)
_x2 = np.array([j for i, j in x_set]).reshape(-1, 1)

# 데이터셋으로 만들겠습니다.
train = tf.data.Dataset.from_tensor_slices(
    ({'x1': _x1, 'x2': _x2}, y)
)
print(next(iter(train)))


# 의도하지 않게 자체 레이어를 만드는 선행학습을 하겠습니다. 해당 내용은 5장에서 확인 가능합니다.
class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.w1 = tf.Variable([[0.01]], trainable=True)

    def call(self, inputs):
        a1 = tf.matmul(inputs, self.w1)
        return a1

# 모델을 만들어보겠습니다.
def build_model():
    # Input Layer에 딕셔너리 키를 넣어서 제대로 들어가게 해줍니다.
    x1 = Input((1,), name='x1')
    x2 = Input((1,), name='x2')
    a1 = MyLayer()(x1)
    a2 = MyLayer()(x2)
    output = tf.keras.layers.Add()([a1, a2])
    model = Model([x1, x2], output)
    return model


model = build_model()
model.summary()

# 모델을 학습시켜보겠습니다.
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mse',
    metrics=['mae'],
)

model.fit(
    train,
    batch_size=32,
    epochs=3,
)

for name, i in zip(['x1', 'x2'], model.trainable_weights):
    print(name, " : ", i.numpy())
"""
결과는 다음과 같이 훌륭하게 나온 것을 확인할 수 있습니다. 
x1  :  [[2.9999995]]
x2  :  [[4.000001]]
"""
