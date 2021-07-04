import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

"""
이번 시간에는 model.fit을 상속받아 custom하게 만드는 방법에 대해서 배워보겠습니다. 
아래와 같이 subclassing 모델을 만들어보겠습니다.
"""

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.h1 = Dense(10, activation='relu')
        self.h2 = Dense(2, activation='softmax')

    def call(self, inputs):
        h = self.h1(inputs)
        h = self.h2(h)
        return h

    # 아래 부분을 구현하면 Custom한 fit으로 만들 수 있습니다.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            result = self(x)
            # 아래는 컴파일한 로스를 적용하는 방법입니다.
            loss = self.compiled_loss(y, result, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradient(zip(gradients, self.trainable_variables))
        # 결과 저장
        self.compiled_metrics.update_state(y, result)
        return {m.name: m.result() for m in self.metrics}