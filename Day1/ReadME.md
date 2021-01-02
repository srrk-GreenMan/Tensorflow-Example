# Day1 

* TF Records
* TF Data & Augmentation 

## TF Records
CIFAR100을 TF Record로 변환해보자!  
기존 예제들이 주로 TFDS등의 예제를 다루지만 실생활에 밀접한 torch ImageFolder처럼 구성하는 방법을 확인해보고 싶었습니다.  
pickle 형태가 아닌 이미지가 들어있는 폴더로 구성되어 있는 CIFAR100을 TF Record로 변환하는 방법에 대해서 다루었습니다.

## TF Data & Augmentation
tf.data에서 주로 사용되는 함수들`# ex) batch, zip, prefetch ... `을 적어두었고 예제들을 적어두었습니다.
Tensorflow에서 Augmentation을 적용하는 방법에 대해서 기술하였습니다. 
