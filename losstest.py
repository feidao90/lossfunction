import tensorflow as tf
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)
tagert = tf.constant(0.)

  # L2正则损失函数(即欧拉损失函数，目标值附近较平滑，收敛性好，距离目标值收敛约慢)
l2_y_vals = tf.square(tagert - x_vals)
l2_y_out = sess.run(l2_y_vals)  #L2:nn.l2_loss()的两倍

# print(sess.run(tf.square(tagert - x_vals)))
# print(sess.run(x_vals))

  # L1正则损失函数(即绝对值损失函数,目标值附近不平滑，收敛性不好)
l1_y_vals = tf.abs(tagert - x_vals)
l1_y_out = sess.run(l1_y_vals)

  #  Pseudo-Huber损失函数（连续、平滑估计，利用L1和L2正则削减极值处的陡峭，目标附近连续）
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1.+tf.square((tagert - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)


delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((tagert - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

# print(sess.run(tf.sqrt((tagert - x_vals)/delta1)) - 1.)

x_vals = tf.linspace(-3., 5., 500)
tagert = tf.constant(1.)
tagerts = tf.fill([500,], 1.)

  # Hinge损失函数，评估支持向量机算法，偶尔用于评估神经网络，计算目标在(-1,1)之间损失

hinge_y_vals = tf.maximum(0., 1. - tf.multiply(tagert, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

  # 两类交叉熵损失函数(Sigmoid cross entropy loss),先通过sigmoid函数转换，再计算交叉熵损失
xentropy_sigmoid_y_vals = -tf.multiply(tagert, tf.log(x_vals)) - tf.multiply((1. -tagert), tf.log(1. - x_vals))
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

  # 加权交叉熵损失函数,Sigmoid交叉熵损失函数加权,对正目标加权
weight = 0.5
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, tagerts,weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

  #Softmax交叉熵损失函数，非归一化结果，针对单个目标分类计算损失，通过softmax函数将输出结果转化为概率分布，然后计算真值概率分布的损失
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)
print(sess.run(softmax_xentropy))

  #稀疏Softmax交叉熵损失喊出，把目标分类true的转化为index
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits,labels=sparse_target_dist)
print(sess.run(sparse_xentropy))

x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label = 'L2 loss')
plt.plot(x_array, l1_y_out, 'r--', label = 'L1 loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label = 'p-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label = 'p-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc = 'lower right', prop = {'size' : 11})
plt.show()

print("aaaa")