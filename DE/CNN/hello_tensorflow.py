import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
hello = tf.constant('Hello, TensorFlow!')
meaning = tf.constant('The Answer to Life, the Universe and Everything is ')

sess    = tf.Session()
msg_op  = sess.run(hello)
mean_op = sess.run(meaning)
print(msg_op)
print(mean_op)

a       = tf.constant(10)
b       = tf.constant(32)
cal_op  = sess.run(a + b)
print(cal_op)

sess.close()

