import tensorflow as tf
import numpy as np

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, linear_warmup_steps, epochs):
        self.initial_learning_rate = initial_learning_rate
        self.linear_warmup_steps = linear_warmup_steps
        self.total_steps = epochs

    def __call__(self, step):
        if step < self.linear_warmup_steps:
            return self.initial_learning_rate * step / self.linear_warmup_steps
        return self.initial_learning_rate * 0.5 * (1 + tf.cos(float(step - self.linear_warmup_steps) / float(self.total_steps - self.linear_warmup_steps) * tf.constant(np.pi)))
