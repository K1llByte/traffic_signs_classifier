import tensorflow as tf

dataset = tf.data.Dataset.range(3)
print(list(dataset.as_numpy_iterator()))
dataset = dataset.shuffle(1, reshuffle_each_iteration=True)
dataset = dataset.repeat(2)
print(list(dataset.as_numpy_iterator()))