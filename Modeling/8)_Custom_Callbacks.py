import tensorflow as tf

"""
이번 시간에는 Custom한 callback을 만들어보겠습니다. 
아래 조건에 해당하는 콜백 함수를 만들어보겠습니다. 

best val acc 모델 저장, val-loss 2번 이상 개선 없으면 stop, 5번 마다 측정  
"""

class CustomCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, patience):
        super(CustomCallbacks, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best = None
        self.best_acc = 0
        self.best_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            acc = logs.get("val_acc")
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_weights = self.model.get_weights()

            loss = logs.get("val_loss")
            if self.best_loss > loss:
                self.best_loss = loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)


