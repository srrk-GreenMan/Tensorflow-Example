import os
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import GlobalAvgPool2D, Dense
import tensorflow_datasets as tfds


# 이번 시간에는 이미 학습된 tf.keras 모델을 사용하여 FineTuning을 하는 방법에 대해서 배워보겠습니다.
# 학습을 위해 일단 데이터를 준비해보겠습니다.

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)


def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (160, 160))
  return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# 모바일넷2 호출해보겠습니다.
# 호출할 때 이미지넷 프리트레인을 사용해보겠습니다.

model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=True
)

model.summary()
"""
Model: "mobilenetv2_1.00_224"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
Conv1_pad (ZeroPadding2D)       (None, 225, 225, 3)  0           input_1[0][0]                    
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 112, 112, 32) 864         Conv1_pad[0][0]                  
__________________________________________________________________________________________________
bn_Conv1 (BatchNormalization)   (None, 112, 112, 32) 128         Conv1[0][0]                      
__________________________________________________________________________________________________
... [중략] ...
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 1280)         0           out_relu[0][0]                   
__________________________________________________________________________________________________
predictions (Dense)             (None, 1000)         1281000     global_average_pooling2d[0][0]   
==================================================================================================
"""

# 모바일넷의 Feature Extraction 부분만 사용해보겠습니다.
# Feature Extraction은 마지막 Dense Layer를 제외한 부분입니다.
# 우선 Input 부터 바꿔보도록 하겠습니다.

# 1. input_shape만 지정하여 진행하는 방법
# ImageNet으로 프리트레인된 웨이트의 input_shape (224, 224, 3)입니다.
# cat and dog의 input_shape는 (160, 160, 3)입니다.

model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)
model.trainable = False
new_model_1 = Sequential([
    model, GlobalAvgPool2D(), Dense(2, activation='softmax')
], name='First Method')
new_model_1.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_160 (Model) (None, 5, 5, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1280)              0         
_________________________________________________________________
dense (Dense)                (None, 2)                 2562      
=================================================================
"""

# 2. Input Layer를 먼저 선언하고 다음에 모델 웨이트를 불러오는 방법

new_input = Input((160, 160, 3))
model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_tensor=new_input
)
model.trainable = False
x = GlobalAvgPool2D()(model.output)
outputs = Dense(2, activation='softmax')(x)
new_model_2 = Model(new_input, outputs, name='Second Method')

new_model_2.summary()
"""
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 160, 160, 3) 0                                            
__________________________________________________________________________________________________
Conv1_pad (ZeroPadding2D)       (None, 161, 161, 3)  0           input_2[0][0]                    
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 80, 80, 32)   864         Conv1_pad[0][0]                  
__________________________________________________________________________________________________
... [중략] ...
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d_2 (Glo (None, 1280)         0           out_relu[0][0]                   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2)            2562        global_average_pooling2d_2[0][0] 
==================================================================================================
"""

# 3. 이미 선언된 모델의 output을 바꿔서 진행하는 방법 -> 난이도 상
"""
참고 사항 
tf.keras에서는 keras 처럼 model.layers.pop()이 먹히지 않음.
There are two issues in your code:
You can't use model.layers.pop() to remove the last layer in the model.
In tf.keras, model.layers will return a shallow copy version of the layers list,
so actually you don't remove that layer, just remove the layer in the return value.
If you want to remove the last dense layer and add your own one,
you should use hidden = Dense(120, activation='relu')(model.layers[-2].output).
model.layers[-1].output means the last layer's output which is the final output,
so in your code, you actually didn't remove any layers.
Thanks.
"""

model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=True, input_shape=(160, 160, 3)
)
model.summary()
feat_ext= Model(model.input, model.get_layer('out_relu').output, name='Third Method')
new_model_3 = Sequential([
    feat_ext,
    GlobalAvgPool2D(),
    Dense(2, activation='softmax')
])
new_model_3.summary()