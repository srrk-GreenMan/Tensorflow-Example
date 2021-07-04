import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN

"""
이번 시간에는 간단한 RNN 구조에 대해서 배워보겠습니다. 
"""

# Embedding Layer는 다음과 같이 사용합니다.
"""
embeddings = tf.keras.layers.Embedding(
    input_dim = 단어 수 , output_dim = 차원 
)
"""

# return sequence가 true면 길이가 3인 sequence가 들어가면 출력도 [o1, o2, o3]으로 나옵니다.
# 그러므로 레이어를 쌓기 위해서는 return_sequence를 True로 두고 진행하세요!
ex = tf.convert_to_tensor([[1.0]*5, [2.0]*5, [3.0]*5])
ex = tf.reshape(ex, [1, 3, 5])
rnn_ = SimpleRNN(2, return_sequences=True)
print(rnn_(ex))
"""
tf.Tensor(
[[[-0.34357     0.9571421 ]
  [-0.79350185  0.99361664]
  [-0.8082363   0.9997333 ]]], shape=(1, 3, 2), dtype=float32)
"""

# return sequence가 False면 길이가 3인 sequence가 들어가면 출력도 3으로 나옵니다.
ex = tf.convert_to_tensor([[1.0]*5, [2.0]*5, [3.0]*5])
ex = tf.reshape(ex, [1, 3, 5])
rnn_ = SimpleRNN(2, return_sequences=False)
print(rnn_(ex))
"""
tf.Tensor([[-0.9998889   0.99940646]], shape=(1, 2), dtype=float32)
"""

# Bidirectional은 다음과 같이 진행합니다.
"""
recurrent_layer = 이미 만들어진 rnn layer Ex LSTM(~)
recurrent_layer = tf.keras.layers.Bidirectional(recurrent_layer)
"""