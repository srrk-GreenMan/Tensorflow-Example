import numpy as np
import tensorflow as tf

"""
이번 시간에는 텐서플로우에 대한 기본적인 기능들에 대해서 배워볼 예정입니다. 

간단한 변수 선언부터 시작해서 다양한 함수들에 대해서 배워보고 에러들을 처리하는 방법에 대해서 배워보겠습니다.
"""

# TF 변수 만들기
# torch.tensor 같은 친구 만들기
"""
torch에서는 간단한 변수를 다음과 같이 만듭니다. 

torch.tensor([1, 2, 3, 4]) 

텐서플로우에서는 다음과 같이 만듭니다. 
"""

temp = tf.Variable([1, 2, 3, 4])
print(temp.shape)

# Numpy tensor에서 tf tensor로 바꾸기
temp = np.ones((15, 15, 3))
temp = tf.convert_to_tensor(temp)
print(temp)

# Tensor Variable 빠르게 만들기
# 1로만 가득찬 (2,3)의 tensor 변수 만들기
temp = tf.fill((2, 3), 1)

"""
변수 타입 변환
"""
a = np.array([1, 2, 3], dtype=np.int32)
a = tf.convert_to_tensor(a)
a_new = tf.cast(a, tf.float32)
print(a, a_new)

"""
텐서 전치하는 방법 & reshape & squeeze
"""
t = tf.random.uniform(shape=(3, 5))
t = tf.transpose(t)
print(t.shape)

t = np.arange(0, 16)
t = tf.convert_to_tensor(t)
print(tf.reshape(t, shape=(4, 2, 2)))

t = tf.zeros((1, 2, 3, 1, 4))
t = tf.squeeze(t, axis=(0, 3))
print(t.shape)

"""
특정 축들에 따라 평균, 합, 표준 편차를 계산하고 싶은 경우
"""
t = np.arange(0, 16)
t = tf.convert_to_tensor(t)
t = tf.reshape(t, (8, 2))
print(tf.math.reduce_sum(t, axis=1))
# tf.Tensor([ 1  5  9 13 17 21 25 29], shape=(8,), dtype=int64)
print(tf.math.reduce_mean(t, axis=0))
# tf.Tensor([7 8], shape=(2,), dtype=int64)

# 표준편차는 아래와 같이
# print(tf.math.reduce_std())

"""
원소별 곱셈, 행렬 곱 
"""
# tf.multiply -> 원소별 곱셈
# tf.matmul -> 행렬 곱

"""
Norm 계산하기 
"""
t = np.arange(0, 16)
t = tf.convert_to_tensor(t)
t = tf.reshape(t, (8, 2))
t = tf.cast(t, tf.float32) # int32는 norm이 계산이 안 됨
norm2 = tf.norm(t, ord=2, axis=1)
print(norm2)

"""
tf.split(), tf.stack(), tf.concat()
"""
# split()
tf.random.set_seed(1)
t = tf.random.uniform((6, ))
print(t.numpy())
# [0.16513085 0.9014813  0.6309742  0.4345461  0.29193902 0.64250207]
t = tf.split(t, num_or_size_splits=3)
for item in t:
    print(item.numpy())
# [0.16513085 0.9014813 ]
# [0.6309742 0.4345461]
# [0.29193902 0.64250207]

# stack
a, b = np.array([1, 2]), np.array([3, 4])
a, b = map(tf.convert_to_tensor, [a, b])
t = tf.stack([a, b], axis=0)
print(t.numpy())
# [[1 2]
# [3 4]]

# concat
a, b = np.array([1, 2]), np.array([3, 4])
a, b = map(tf.convert_to_tensor, [a, b])
t = tf.concat([a, b], axis=0)
print(t.numpy())
# [1 2 3 4]

"""
그라디언트에 대한 이해 
"""
# 그라디언트가 저장되지 않는 친구로 만들고 싶다면 다음과 같이 만들면 됩니다.

# 1. trainable=False
temp2 = tf.Variable([1, 2, 3, 4], trainable=False)

temp3 = tf.constant([1, 2, 3, 4])

print(temp.trainable, temp2.trainable)

# 그라디언트를 꺼내다가 쓰는 법은 다음과 같습니다.

grad_ex = tf.Variable([4.])
print(grad_ex)
# 주의 할 점은 int32로 gradient를 계산하는 경우 None이 뜰 수 있습니다.
# 변수가 꼭 tf.float32임을 확인해주세요!
with tf.GradientTape() as t:
    t.watch(grad_ex)
    y = grad_ex * grad_ex
dy_dx = t.gradient(y, grad_ex)
print('Caculate Result: ', y, 'grad: ', dy_dx)


