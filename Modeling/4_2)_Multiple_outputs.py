import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense

"""
모델을 만들면 하나의 input, output을 선정하지 않는 경우가 있습니다. 이번 시간에는 그러한 문제들을 해결하는 방법에 대해서 배워보겠습니다.
"""

# Multiple outputs
# 이전에 진행했던 함수 식에 대해서 다른 방법으로 접근해보겠습니다.
# x1, x2 계수들을 레이어를 분리하여 따로 추정해보겠습니다.
# 이번엔 y1 = 3 * x1, y2 = 4 * x2로 추정해보겠습니다.

x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)

y1 = np.array([i * 3 for i in x1]).reshape(-1, 1)
y2 = np.array([j * 4 for j in x2]).reshape(-1, 1)

_x1 = np.array(x1).reshape(-1, 1)
_x2 = np.array(x2).reshape(-1, 1)

# 데이터셋으로 만들겠습니다.
train = tf.data.Dataset.from_tensor_slices(
    ({'x1': _x1, 'x2': _x2}, {'y1': y1, 'y2': y2})
)

# 모델을 만들어보겠습니다.
def build_model():
    # Input Layer에 딕셔너리 키를 넣어서 제대로 들어가게 해줍니다.
    x1 = Input((1,), name='x1')
    x2 = Input((1,), name='x2')
    # 마찬가지로 output 역시 이름을 맞춰줍시다.
    a1 = Dense(1, name='y1')(x1)
    a2 = Dense(1, name='y2')(x2)
    model = Model([x1, x2], [a1, a2])
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
