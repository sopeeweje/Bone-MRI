from keras import optimizers
from keras.callbacks import LearningRateScheduler
import math

OPTIMIZERS = {
    "sgd-01-0.9": lambda: optimizers.SGD(lr=0.01, momentum=0.9),
    "sgd-001-0.9": lambda: optimizers.SGD(lr=0.001, momentum=0.9),
    "sgd-0001-0.9": lambda: optimizers.SGD(lr=0.0001, momentum=0.9),
    "sgd-01-0.9-nesterov": lambda: optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    "sgd-001-0.9-nesterov": lambda: optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
    "sgd-0001-0.9-nesterov": lambda: optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    "sgd-00001-0.9-nesterov": lambda: optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True),
    "sgd-e6-0.9-nesterov": lambda: optimizers.SGD(lr=1e-6, momentum=0.9, nesterov=True),
    "sgd-01-0.9-expdecay": lambda: optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01/500),
    "adam": lambda: "adam",
    "nadam": lambda: "nadam",
    "clr": lambda: clr
}

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)
#Exponential decay schedules
#expd_01 = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=500, decay_rate=0.001)
#tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None)
#decayed_learning_rate = initial_learning_rate * decay_rate ^ (step / decay_steps)

#optimizer = optimizers.SGD(learning_rate=lr_schedule)

#Piecewise decay schedules
#https://keras.io/api/optimizers/learning_rate_schedules/piecewise_constant_decay/
#tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values, name=None)
