import tensorflow as tf

model = tf.saved_model.load('../../../savedmodeldir/')
print(list(model.signatures.keys()))
