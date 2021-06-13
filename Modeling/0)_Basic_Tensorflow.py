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


