# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:02:32 2021

@author: lenovo
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from eager_lbfgs import lbfgs, Struct
from pyDOE import lhs


layer_sizes = [3, 20, 20, 20, 20, 20, 20, 20, 3]
sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))


# L-BFGS weight getting and setting from https://github.com/pierremtb/PINNs-TF2.0

def set_weights(model, w, sizes_w, sizes_b):  # 重新设置参数

    for i, layer in enumerate(model.layers[1:len(sizes_w) + 1]):
        start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
        end_weights = sum(sizes_w[:i + 1]) + sum(sizes_b[:i])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[i] / sizes_b[i])
        weights = tf.reshape(weights, [w_div, sizes_b[i]])
        biases = w[end_weights:end_weights + sizes_b[i]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)


def get_weights(model):
    w = []
    for layer in model.layers[1:len(sizes_w) + 1]:
        weights_biases = layer.get_weights()
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)
    w = tf.convert_to_tensor(w)
    return w

def xavier_init(layer_sizes):
    in_dim = layer_sizes[0]
    out_dim = layer_sizes[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net(layer_sizes):

    input_tensor = keras.Input(shape=(layer_sizes[0],))

    hide_layer_list = []
    flag = True
    for width in layer_sizes[1:-1]:
        if flag:
            x = layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal")(input_tensor)
            flag = False
        else:
            x = layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal")(x)
    output_tensor = layers.Dense(layer_sizes[-1], activation=None,kernel_initializer="glorot_normal")(x)
    print("xxxxxxxxxxxxxx")
    output0 = output_tensor[:, 0:1]
    output1 = output_tensor[:, 1:2]
    output2 = output_tensor[:, 2:3]

    model_output = keras.models.Model(input_tensor, [output0, output1, output2])

    return model_output

# initialize the NN
u_model = neural_net(layer_sizes)
# view the NN
u_model.summary()


# define the loss
def loss(x_f_batch, y_f_batch, t_f_batch, xb, yb, tb, ub, vb, weight_ub,  weight_fu):

    f_u_pred, f_v_pred, div_pred = f_model(x_f_batch, y_f_batch, t_f_batch)


    u_pred, v_pred, p_pred = u_model(tf.concat([xb, yb, tb], 1))
    mse_b = 100*weight_ub*(tf.reduce_sum(tf.square(u_pred - ub)) + tf.reduce_sum(tf.square(v_pred - vb)))
    mse_f = weight_fu*(tf.reduce_sum(tf.square(f_u_pred)) + tf.reduce_sum(tf.square(f_v_pred)) + tf.reduce_sum(tf.square(div_pred)))

    return mse_b + mse_f, mse_b, mse_f

@tf.function
def f_model(x, y, t):
    u, v, p = u_model(tf.concat([x, y, t],1))

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]

    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    div = u_x + v_y
    c1 = tf.constant(0.01, dtype=tf.float32)
    f_u = u_t + u*u_x + v*u_y + p_x -c1*(u_xx + u_yy) - ((np.pi)*tf.cos(np.pi*x)*tf.cos(np.pi*y)*tf.sin(t) - tf.cos(np.pi*y)*(tf.sin(np.pi*x))**2*tf.sin(np.pi*y)*tf.cos(t) + \
    c1*(2*(np.pi)**2*(tf.cos(np.pi*x))**2*tf.cos(np.pi*y)*tf.sin(np.pi*y)*tf.sin(t) - 6*(np.pi)**2*tf.cos(np.pi*y)*(tf.sin(np.pi*x))**2*tf.sin(np.pi*y)*tf.sin(t)) - \
    tf.cos(np.pi*x)*tf.sin(np.pi*x)*(tf.sin(np.pi*y))**2*tf.sin(t)*(np.pi*(tf.cos(np.pi*y))**2*(tf.sin(np.pi*x))**2*tf.sin(t) - np.pi*(tf.sin(np.pi*x))**2*(tf.sin(np.pi*y))**2*tf.sin(t)) +\
    2*np.pi*tf.cos(np.pi*x)*(tf.cos(np.pi*y))**2*(tf.sin(np.pi*x))**3*(tf.sin(np.pi*y))**2*(tf.sin(t))**2)

    f_v = v_t + u*v_x + v*v_y + p_y - c1*(v_xx + v_yy) - (tf.cos(np.pi*x)*tf.sin(np.pi*x)*(tf.sin(np.pi*y))**2*tf.cos(t) - np.pi*tf.sin(np.pi*x)*tf.sin(np.pi*y)*tf.sin(t) - \
    c1*(2*(np.pi)**2*tf.cos(np.pi*x)*(tf.cos(np.pi*y))**2*tf.sin(np.pi*x)*tf.sin(t) - 6*(np.pi)**2*tf.cos(np.pi*x)*tf.sin(np.pi*x)*(tf.sin(np.pi*y))**2*tf.sin(t)) -\
    tf.cos(np.pi*y)*(tf.sin(np.pi*x))**2*tf.sin(np.pi*y)*tf.sin(t)*(np.pi*(tf.cos(np.pi*x))**2*(tf.sin(np.pi*y))**2*tf.sin(t) -\
    np.pi*(tf.sin(np.pi*x))**2*(tf.sin(np.pi*y))**2*tf.sin(t)) + 2*np.pi*(tf.cos(np.pi*x))**2*tf.cos(np.pi*y)*(tf.sin(np.pi*x))**2*(tf.sin(np.pi*y))**3*(tf.sin(t))**2)

    return f_u, f_v, div

@tf.function
def u_x_model(x, y, t):
    u, v, w = u_model(tf.concat([x, y, t], 1))
    return u, v, w


@tf.function
def grad(u_model, x_f_batch, y_f_batch, t_f_batch, xb_batch, yb_batch, tb_batch, ub_batch, vb_batch, weight_ub,
         weight_fu):
    with tf.GradientTape(persistent=True) as tape:

        loss_value, mse_b, mse_f = loss(x_f_batch, y_f_batch, t_f_batch, xb_batch, yb_batch, tb_batch, ub_batch,
                                        vb_batch, weight_ub, weight_fu)
        grads = tape.gradient(loss_value, u_model.trainable_variables)

        grads_ub = tape.gradient(loss_value, weight_ub)

        grads_fu = tape.gradient(loss_value, weight_fu)

    return loss_value, mse_b, mse_f, grads, grads_ub, grads_fu


def fit(x_f, y_f, t_f, xb, yb, tb, ub, vb, weight_ub, weight_fu, u_exact1, v_exact1, p_exact1, X_star, tf_iter, tf_iter2,
        newton_iter1, newton_iter2):

    batch_sz = N_f
    n_batches = N_f // batch_sz

    start_time = time.time()

    tf_optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=.99)
    tf_optimizer_weights = tf.keras.optimizers.Adam(lr=0.003, beta_1=.99)
    tf_optimizer_u = tf.keras.optimizers.Adam(lr=0.03, beta_1=.99)

    tf.print(f"weight_ub: {weight_ub}  weight_fu: {weight_fu}")
    print("starting Adam training")

    a = np.random.rand(1000)
    loss_history = list(a)
    MSE_b0 = list(a)
    MSE_f0 = list(a)

    MSE_b1 = []
    MSE_f1 = []

    weightu = []
    weightf = []
    # For mini-batch (if used)
    for epoch in range(tf_iter):
        for i in range(n_batches):
            xb_batch = xb
            yb_batch = yb
            tb_batch = tb
            ub_batch = ub
            vb_batch = vb

            x_f_batch = x_f[i * batch_sz:(i * batch_sz + batch_sz), ]
            y_f_batch = y_f[i * batch_sz:(i * batch_sz + batch_sz), ]
            t_f_batch = t_f[i * batch_sz:(i * batch_sz + batch_sz), ]

            loss_value, mse_b, mse_f, grads, grads_ub, grads_fu = grad(u_model, x_f_batch, y_f_batch, t_f_batch,
                                                                       xb_batch, yb_batch,
                                                                       tb_batch, ub_batch, vb_batch, weight_ub,
                                                                       weight_fu)

            tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
            MSE_b0.append(mse_b)
            MSE_f0.append(mse_f)

            loss_history.append(loss_value)
            
            if loss_history[-1] < loss_history[-2] and loss_history[-2] < loss_history[-3] and loss_history[-1] < \
                    loss_history[-10]:
                tf_optimizer_weights.apply_gradients(zip([-grads_fu], [weight_fu]))
                tf_optimizer_u.apply_gradients(zip([-grads_ub], [weight_ub]))

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_b  {mse_b}  mse_f: {mse_f}   total loss: {loss_value}")

            wu = weight_ub.numpy()
            wf = weight_fu.numpy()

            MSE_b1.append(mse_b)
            MSE_f1.append(mse_f)

            weightu.append(wu)
            weightf.append(wf)

            start_time = time.time()
    tf.print(f"weight_ub: {weight_ub}  weight_fu: {weight_fu}")
    u_pred, v_pred, p_pred = predict(X_star)
    error_u = np.linalg.norm(u_exact1 - u_pred, 2) / np.linalg.norm(u_exact1, 2)
    print('Error u: %e' % (error_u))
    error_v = np.linalg.norm(v_exact1 - v_pred, 2) / np.linalg.norm(v_exact1, 2)
    print('Error v: %e' % (error_v))
    print("Starting L-BFGS training")

    loss_and_flat_grad = get_loss_and_flat_grad(x_f_batch, y_f_batch, t_f_batch, xb_batch, yb_batch, tb_batch, ub_batch,
                                                vb_batch, weight_ub, weight_fu)

    lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter1, learningRate=0.8)

    u_pred, v_pred, p_pred = predict(X_star)
    error_u = np.linalg.norm(u_exact1 - u_pred, 2) / np.linalg.norm(u_exact1, 2)
    print('Error u: %e' % (error_u))
    error_v = np.linalg.norm(v_exact1 - v_pred, 2) / np.linalg.norm(v_exact1, 2)
    print('Error v: %e' % (error_v))

    lbfgs(loss_and_flat_grad,
          get_weights(u_model),
          Struct(), maxIter=newton_iter2, learningRate=0.8)

    return MSE_b1, MSE_f1,  weightu, weightf

# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, y_f_batch, t_f_batch, xb_batch, yb_batch,
                           tb_batch, ub_batch, vb_batch,weight_ub, weight_fu):
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _ = loss(x_f_batch, y_f_batch, t_f_batch, xb_batch, yb_batch, tb_batch, ub_batch, vb_batch, weight_ub, weight_fu)
        grad = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        # print(loss_value, grad_flat)
        return loss_value, grad_flat

    return loss_and_flat_grad


def predict(X_star):
    X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
    u_star, v_star, p_star = u_x_model(X_star[:, 0:1], X_star[:, 1:2], X_star[:, 2:3])
    return u_star.numpy(), v_star.numpy(), p_star.numpy()


N_f = 10000
Nu1 = 200

weight_ub = tf.Variable([1.0], dtype=tf.float32)
weight_fu = tf.Variable([1.0], dtype=tf.float32)

x1 = (np.linspace(0, 1, 32)).flatten()[:, None]
y1 = (np.linspace(0, 1, 32)).flatten()[:, None]
t1 = (np.linspace(0, 1, 20)).flatten()[:, None]

ttt1, ttt0 = np.meshgrid(x1, y1)

tt1 = np.concatenate(([ttt1.flatten()[:, None], ttt0.flatten()[:, None], np.zeros((x1.shape[0] * y1.shape[0], 1))]), axis=1)
x_1t = np.array([tt1[:, 0]]).T
y_1t = np.array([tt1[:, 1]]).T
t_1t = np.array([tt1[:, 2]]).T
ut1 = -np.sin(t_1t) * np.sin(np.pi * x_1t) * np.sin(np.pi * x_1t) * np.sin(np.pi * y_1t) * np.cos(np.pi * y_1t)
vt1 = np.sin(t_1t) * np.sin(np.pi * x_1t) * np.cos(np.pi * x_1t) * np.sin(np.pi * y_1t) * np.sin(np.pi * y_1t)

yyy1, yyy0 = np.meshgrid(x1, t1)

yy1 = np.concatenate(
    ([yyy1.flatten()[:, None], np.min(y1) * np.ones((x1.shape[0] * t1.shape[0], 1)), yyy0.flatten()[:, None]]), axis=1)
x_1y = np.array([yy1[:, 0]]).T
y_1y = np.array([yy1[:, 1]]).T
t_1y = np.array([yy1[:, 2]]).T
uy1 = -np.sin(t_1y) * np.sin(np.pi * x_1y) * np.sin(np.pi * x_1y) * np.sin(np.pi * y_1y) * np.cos(np.pi * y_1y)
vy1 = np.sin(t_1y) * np.sin(np.pi * x_1y) * np.cos(np.pi * x_1y) * np.sin(np.pi * y_1y) * np.sin(np.pi * y_1y)

yy2 = np.concatenate(
    ([yyy1.flatten()[:, None], np.max(y1) * np.ones((x1.shape[0] * t1.shape[0], 1)), yyy0.flatten()[:, None]]), axis=1)
x_2y = np.array([yy2[:, 0]]).T
y_2y = np.array([yy2[:, 1]]).T
t_2y = np.array([yy2[:, 2]]).T
uy2 = -np.sin(t_2y) * np.sin(np.pi * x_2y) * np.sin(np.pi * x_2y) * np.sin(np.pi * y_2y) * np.cos(np.pi * y_2y)
vy2 = np.sin(t_2y) * np.sin(np.pi * x_2y) * np.cos(np.pi * x_2y) * np.sin(np.pi * y_2y) * np.sin(np.pi * y_2y)


xxx1, xxx0 = np.meshgrid(y1, t1)

xx1 = np.concatenate(
    ([np.min(x1) * np.ones((y1.shape[0] * t1.shape[0], 1)), xxx1.flatten()[:, None], xxx0.flatten()[:, None]]), axis=1)
x_1x = np.array([xx1[:, 0]]).T
y_1x = np.array([xx1[:, 1]]).T
t_1x = np.array([xx1[:, 2]]).T
ux1 = -np.sin(t_1x) * np.sin(np.pi * x_1x) * np.sin(np.pi * x_1x) * np.sin(np.pi * y_1x) * np.cos(np.pi * y_1x)
vx1 = np.sin(t_1x) * np.sin(np.pi * x_1x) * np.cos(np.pi * x_1x) * np.sin(np.pi * y_1x) * np.sin(np.pi * y_1x)

xx2 = np.concatenate(
    ([np.max(x1) * np.ones((y1.shape[0] * t1.shape[0], 1)), xxx1.flatten()[:, None], xxx0.flatten()[:, None]]), axis=1)
x_2x = np.array([xx2[:, 0]]).T
y_2x = np.array([xx2[:, 1]]).T
t_2x = np.array([xx2[:, 2]]).T
ux2 = -np.sin(t_2x) * np.sin(np.pi * x_2x) * np.sin(np.pi * x_2x) * np.sin(np.pi * y_2x) * np.cos(np.pi * y_2x)
vx2 = np.sin(t_2x) * np.sin(np.pi * x_2x) * np.cos(np.pi * x_2x) * np.sin(np.pi * y_2x) * np.sin(np.pi * y_2x)

X_u1 = np.vstack([tt1, yy1, yy2, xx1, xx2])
u1 = np.vstack([ut1, uy1, uy2, ux1, ux2])
v1 = np.vstack([vt1, vy1, vy2, vx1, vx2])

idx_1 = np.random.choice(X_u1.shape[0], Nu1, replace=False)
X_u_train = X_u1[idx_1, :]
u_train = u1[idx_1, :]
v_train = v1[idx_1, :]

X1, Y1, T1 = np.meshgrid(x1, y1, t1)
#    Exact = np.sin(np.pi*X)*np.sin(np.pi*T)*np.sin(np.pi*Z)  #100*100*100
U_exact1 = -np.sin(T1) * np.sin(np.pi * X1) * np.sin(np.pi * X1) * np.sin(np.pi * Y1) * np.cos(np.pi * Y1)
V_exact1 = np.sin(T1) * np.sin(np.pi * X1) * np.cos(np.pi * X1) * np.sin(np.pi * Y1) * np.sin(np.pi * Y1)
P_exact1 = np.sin(T1) * np.sin(np.pi * X1) * np.cos(np.pi * Y1)

X_star1 = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None], T1.flatten()[:, None]))
x_star1 = np.array([X_star1[:, 0]]).T
y_star1 = np.array([X_star1[:, 1]]).T
t_star1 = np.array([X_star1[:, 2]]).T

u_exact1 = -np.sin(t_star1) * np.sin(np.pi * x_star1) * np.sin(np.pi * x_star1) * np.sin(np.pi * y_star1) * np.cos(np.pi * y_star1)
v_exact1 = np.sin(t_star1) * np.sin(np.pi * x_star1) * np.cos(np.pi * x_star1) * np.sin(np.pi * y_star1) * np.sin(np.pi * y_star1)
p_exact1 = np.sin(t_star1) * np.sin(np.pi * x_star1) * np.cos(np.pi * y_star1)

lb1 = X_star1.min(0)
ub1 = X_star1.max(0)

X_f_train11 = lb1 + (ub1 - lb1) * lhs(3, N_f)
X_f = np.vstack((X_f_train11, X_u_train))

xb = tf.cast(X_u_train[:, 0:1], dtype=tf.float32)
yb = tf.cast(X_u_train[:, 1:2], dtype=tf.float32)
tb = tf.cast(X_u_train[:, 2:3], dtype=tf.float32)
ub = tf.cast(u_train[:, 0:1], dtype=tf.float32)
vb = tf.cast(v_train[:, 0:1], dtype=tf.float32)


lb = X_star1.min(0)
rb = X_star1.max(0)

x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
y_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)
t_f = tf.convert_to_tensor(X_f[:, 2:3], dtype=tf.float32)

start_time = time.time()
MSE_b1, MSE_f1, weightu, weightf = fit(x_f, y_f, t_f, xb, yb, tb, ub, vb, weight_ub, weight_fu, u_exact1, v_exact1, p_exact1, X_star1, tf_iter=10000, tf_iter2=1000, newton_iter1=5000,newton_iter2=15000)


elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

u_pred, v_pred, p_pred = predict(X_star1)

U_pred = u_pred.reshape((x1.shape[0], y1.shape[0], t1.shape[0]))
V_pred = v_pred.reshape((x1.shape[0], y1.shape[0], t1.shape[0]))
P_pred = p_pred.reshape((x1.shape[0], y1.shape[0], t1.shape[0]))

error_uu = np.abs(u_exact1 - u_pred)
error_vv = np.abs(v_exact1 - v_pred)
error_pp = np.abs(p_exact1 - p_pred)

error_u = np.linalg.norm(u_exact1 - u_pred, 2) / np.linalg.norm(u_exact1, 2)
print('Error u: %e' % (error_u))

error_v = np.linalg.norm(v_exact1 - v_pred, 2) / np.linalg.norm(v_exact1, 2)
print('Error v: %e' % (error_v))

error_p = np.linalg.norm(p_exact1 - p_pred, 2) / np.linalg.norm(p_exact1, 2)
print('Error p: %e' % (error_p))

dataNewNS = 'D://NS_hisyory.mat'
scipy.io.savemat(dataNewNS, {'w_MSE_b': MSE_b1, 'w_MSE_f': MSE_f1, 'weight_u': weightu,
                  'weight_f': weightf, 'U_pred': U_pred, 'V_pred': V_pred, 'P_pred': P_pred})
