import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.callbacks import Callback

class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25., reduce_lr_patience=10, reduce_lr_factor=0.1):
        super().__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.steps = steps
        self.phase_1_steps = steps * phase_1_pct
        self.phase_2_steps = steps - self.phase_1_steps
        self.lr_min = lr_max / self.div_factor
        self.final_lr = lr_max / (self.div_factor * 1e4)
        self.mom_min = mom_min
        self.mom_max = mom_max
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.phase = 0
        self.step = 0
        
        self.phases = [
            [CosineAnnealer(self.lr_min, self.lr_max, self.phase_1_steps), CosineAnnealer(self.mom_max, self.mom_min, self.phase_1_steps)],
            [CosineAnnealer(self.lr_max, self.final_lr, self.phase_2_steps), CosineAnnealer(self.mom_min, self.mom_max, self.phase_2_steps)]
        ]
        
        self.current_lr = None
        self.current_momentum = None
        self.lr_history = []
        self.momentum_history = []
        self.reduce_lr_counter = 0
        self.lowest_loss = float('inf')

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0
        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lr_history.append(self.current_lr)
        self.momentum_history.append(self.current_momentum)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.lowest_loss:
            self.lowest_loss = val_loss
            self.reduce_lr_counter = 0
        else:
            self.reduce_lr_counter += 1
            if self.reduce_lr_counter >= self.reduce_lr_patience:
                old_lr_min = self.lr_min
                old_final_lr = self.final_lr
                self.update_phases()
                self.reduce_lr_counter = 0
                print(f"\nReducing learning rate at epoch {epoch + 1} - Lowest loss: {self.lowest_loss:.4f}")
                print(f"Old lr_min: {old_lr_min:.6f}, New lr_min: {self.lr_min:.6f}")
                print(f"Old final_lr: {old_final_lr:.10f}, New final_lr: {self.final_lr:.10f}")

                
    def update_phases(self):
        # Update the lr_min and final_lr values
        self.lr_max *= self.reduce_lr_factor
        self.lr_min = self.lr_max / self.div_factor
        self.final_lr = self.lr_max / (self.div_factor * 1e4)
        
        # Update the learning rate schedules in the phases array
        self.phases[0][0] = CosineAnnealer(self.lr_min, self.lr_max, self.phase_1_steps)
        self.phases[1][0] = CosineAnnealer(self.lr_max, self.final_lr, self.phase_2_steps)
                                           
    def on_train_end(self, logs=None):
        print(f"Lowest loss achieved: {self.lowest_loss:.4f}")

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            self.current_lr = lr
        except AttributeError:
            pass

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
            self.current_momentum = mom
        except AttributeError:
            pass

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot_lr_momentum(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.lr_history)
        plt.title('Learning Rate')
        plt.xlabel('Batch Iterations')
        plt.ylabel('LR')
        plt.subplot(1, 2, 2)
        plt.plot(self.momentum_history)
        plt.title('Momentum')
        plt.xlabel('Batch Iterations')
        plt.ylabel('Momentum')
        plt.show()
