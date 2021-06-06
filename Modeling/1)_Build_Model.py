import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input

"""
간단한 LeNet 구조와 비슷한 친구를 3 가지 방법으로 구현해보겠습니다. 
LeNet이란? - https://my-coding-footprints.tistory.com/97
"""

# 1. 순차형 방법
# 우선 필요한 레이어들을 불러오겠습니다.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model_sequential = Sequential()
input_shape = (28, 28, 1)  # 텐서플로우의 차원 순서는 (W, H, C) 입니다.
num_filters = 32
kernel_size = (5, 5)
pool_size = (2, 2)
num_classes = 10

# 모델 만들기 시작!
# 첫 번째 레이어에는 input_shape 변수 넣어서 compile -> fit 을 쉽게 합니다.
# input_shape 변수가 없다하더라도 모델을 만들 수 있지만 파라메터 개수 등을 계산하기 어렵습니다.
# input_shape 변수를 넣는 방법은 여러가지 입니다. 그 중 하나는 첫 번째 레이어에 다음과 같이 집어넣는 것입니다.
# 1st Layer
model_sequential.add(
    Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='valid',  # padding이 valid이면 input 차원과 output 차원이 다릅니다.
        activation='relu',
        input_shape=input_shape
    )
)
# 2nd Layer
model_sequential.add(
    Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='same',  # padding이 valid이면 input 차원과 output 차원이 같습니다.
        activation='relu'
    )
)
# 3rd Layer
model_sequential.add(
    MaxPooling2D(pool_size=pool_size)
)
# Dropout Layer
model_sequential.add(
    Dropout(0.5)
)
# Flatten Layer
model_sequential.add(
    Flatten()
)
# Last Layer
model_sequential.add(
    Dense(units=num_classes, activation='softmax')
)

# 모델을 다 만들어보았습니다. 한 번 확인해볼까요?
print(model_sequential.summary())
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 24, 24, 32)        832       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        25632     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                46090     
=================================================================
Total params: 72,554
Trainable params: 72,554
Non-trainable params: 0
_________________________________________________________________
"""


# 2. 함수형 방법
# 복잡한 함수들을 만들기에 적절한 방법입니다. 아래와 같은 함수를 만들어서 모델을 만듭니다

def build_functional_model():
    # 입력을 다음과 같이 지정해줍니다. 혹은 앞서 했던 예제처럼에 첫 번째 레이어의 input_shape 변수 넣는 방법도 있습니다.
    inputs = Input(input_shape)
    # 1st Layer
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='valid',
        activation='relu'
    )(inputs)

    # 2nd Layer
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu'
    )(x)

    # 3rd Layer
    x = MaxPooling2D(pool_size=pool_size)(x)

    # Dropout Layer
    x = Dropout(0.5)(x)

    # Flatten Layer
    x = Flatten()(x)

    # Last Layer

    outputs = Dense(
        units=num_classes,
        activation='softmax'
    )(x)

    # 다 만든 후 그래프의 인풋과 아웃풋을 Model에 넣어 모델을 만듭니다.
    model = Model(inputs, outputs)
    return model


# 만든 모델을 확인해볼까요?
model_functional = build_functional_model()
print(model_functional.summary())
"""
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        832       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 32)        25632     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                46090     
=================================================================
Total params: 72,554
Trainable params: 72,554
Non-trainable params: 0
_________________________________________________________________
"""

# 3. 서브 클래싱 방법


class SubClassModel(Model):
    def __init__(self):
        super(SubClassModel, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.input_shapes = input_shape

        # 1st Layer
        self.conv1 = Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding='valid',
            activation='relu',
            input_shape=self.input_shapes
        )
        # 2nd Layer
        self.conv2 = Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation='relu'
        )
        # 3rd Layer
        self.mp_layer = MaxPooling2D(
            pool_size=self.pool_size
        )
        # Dropout Layer
        self.drop = Dropout(0.5)
        # Flatten Layer
        self.flatten = Flatten()
        # Last Layer
        self.dense = Dense(
            units=num_classes,
            activation='softmax'
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.mp_layer(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


model_subclassing = SubClassModel()
model_subclassing.build((None, 28, 28, 1))
print(model_subclassing.summary())
"""
Model: "sub_class_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            multiple                  832       
_________________________________________________________________
conv2d_5 (Conv2D)            multiple                  25632     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 multiple                  0         
_________________________________________________________________
dropout_2 (Dropout)          multiple                  0         
_________________________________________________________________
flatten_2 (Flatten)          multiple                  0         
_________________________________________________________________
dense_2 (Dense)              multiple                  46090     
=================================================================
Total params: 72,554
Trainable params: 72,554
Non-trainable params: 0
_________________________________________________________________
"""