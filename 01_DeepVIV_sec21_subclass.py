import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# progress bar for model.fit function
import tensorflow_addons as tfa
import tqdm

# set random seed for reproduction
tf.random.set_seed(42)

# initialize tqdm callback
tqdm_callback = tfa.callbacks.TQDMProgressBar(
    leave_epoch_progress=True,
    leave_overall_progress=True,
    show_epoch_progress=False,
    show_overall_progress=True,
)

VIVdata = loadmat("VIV_displacement_lift_drag.mat")

x = VIVdata["eta_structure"]
f = VIVdata["lift_structure"]
t = VIVdata["t_structure"]

N_train = t.shape[0]  # how much data ?

# time series plots
# plt.figure(figsize=[14, 5])
# plt.subplot(121)
# plt.plot(t, x, "k", linewidth=3)
# plt.legend("x")
# plt.xlabel("t")
# plt.grid()

# plt.subplot(122)
# plt.plot(t, f, "k", linewidth=3)
# plt.legend("f")
# plt.xlabel("t")
# plt.grid()
# plt.show()

# -- classes

# FUNCIONA para treino apenas da 1a parte
# class tToX(tf.keras.Model):
#     """
#     Implements the first block: mapping time (the independent variable) to x (the state)
#     """

#     def __init__(self, name="tToX", **kwargs):
#         super(tToX, self).__init__(name=name, **kwargs)

#         self.layerWidth = 10 * [32]
#         self.actFcn = tf.sin

#         # instantiates the model
#         self.model = tf.keras.Sequential()

#         for nNeurons in self.layerWidth:
#             self.model.add(tf.keras.layers.Dense(nNeurons, activation=self.actFcn))
#         self.model.add(tf.keras.layers.Dense(1))

#     def call(self, t):
#         x = self.model(t)
#         return x


############################################################################################
# NÃO FUNCIONA c/ 2 layers
# class tToX(
#     tf.keras.layers.Layer
# ):  # Implements the first block: mapping time (the independent variable) to x (the state)
#     def __init__(self, name="tToX", **kwargs):
#         super(tToX, self).__init__(name=name, **kwargs)

#         self.layerWidth = 10 * [32]
#         self.actFcn = tf.sin

#         # instantiates all layers recursively
#         self.denseList = []
#         for nNeurons in self.layerWidth:
#             self.denseList.append(
#                 tf.keras.layers.Dense(nNeurons, activation=self.actFcn)
#             )  # hidden layers

#         self.denseList.append(tf.keras.layers.Dense(1))  # output layer

#     def call(self, t):
#         x = t
#         for nLayers in range(self.denseList.__len__()):
#             x = self.denseList[nLayers](t)
#         return x


# class xToF(tf.keras.layers.Layer):  # Implements the second block: maps the ODE
#     def __init__(self, name="xToF", **kwargs):
#         super(xToF, self).__init__(name=name, **kwargs)

#         self.b = tf.Variable(0.05, name="b", trainable=True, dtype=tf.float32)
#         self.k = tf.Variable(2.0, name="k", trainable=True, dtype=tf.float32)

#     def call(self, x, xp, xpp):

#         # calculate the EOM
#         f = 2.0 * xpp + self.b * xp + self.k * x

#         return f


# class PINN(tf.keras.Model):  # Combines the 2 blocks for PINN end-to-end training
#     def __init__(self, name="PINN", **kwargs):
#         super(PINN, self).__init__(name=name, **kwargs)

#         self.tToX = tToX()
#         self.xToF = xToF()

#     def call(self, t):

#         # dont forget to tape the gradients
#         with tf.GradientTape(
#             persistent=True  # persistent for 2nd order derivative
#         ) as tape:
#             tape.watch(t)

#             x = tToX(t)

#             # part 2a: calculate the gradients of x wrt t
#             xp = tape.gradient(x, t)
#             xpp = tape.gradient(xp, t)

#             # part 2b: calculate the EOM
#             f = xToF(x, xp, xpp)

#         return x, f

# Implements the first block: mapping time (the independent variable) to x (the state)
class tToXandF(tf.keras.layers.Layer):
    def __init__(self, layerWidth, actFcn, name="tToX", **kwargs):
        super(tToXandF, self).__init__(name=name, **kwargs)

        self.layerWidth = layerWidth
        self.actFcn = actFcn

        # instantiates all layers recursively
        self.denseList = []
        for nNeurons in self.layerWidth:
            self.denseList.append(
                tf.keras.layers.Dense(nNeurons, activation=self.actFcn)
            )  # hidden layers

        self.denseList.append(tf.keras.layers.Dense(1))  # output layer

        self.b = tf.Variable(0.05, name="b", trainable=True, dtype=tf.float32)
        self.k = tf.Variable(2.0, name="k", trainable=True, dtype=tf.float32)

    def call(self, t):

        with tf.GradientTape(
            persistent=True  # persistent for 2nd order derivative
        ) as tape:

            tape.watch(t)
            # part 1: calculate  t->x
            for nLayers in range(self.denseList.__len__()):
                if nLayers == 0:
                    x = self.denseList[nLayers](t)
                else:
                    x = self.denseList[nLayers](x)

            # part 2a: calculate the gradients of x wrt t
            xp = tape.gradient(x, t)
            xpp = tape.gradient(xp, t)

            # part 2b: calculate the EOM
            f = 2.0 * xpp + self.b * xp + self.k * x

        return x, f


# Combines the 2 blocks for PINN end-to-end training
class PINN(tf.keras.Model):
    def __init__(self, layerWidth=10 * [32], actFcn=tf.sin, name="PINN", **kwargs):
        super(PINN, self).__init__(name=name, **kwargs)

        self.tToXandF = tToXandF(layerWidth=layerWidth, actFcn=actFcn)

    def call(self, t):
        return self.tToXandF(t)


# mdl = PINN()  # default parameters are the same as DeepVIV paper
mdl = PINN(layerWidth=4 * [16], actFcn="sigmoid")

lrvec = np.array([1e-3, 1e-4, 1e-5, 1e-6])
epvec = np.array([2e3, 3e3, 3e3, 2e3], dtype="int32")

nTrain = lrvec.shape[0]
for i in range(nTrain):
    print("Learning rate:", lrvec[i])
    mdl.compile(
        tf.keras.optimizers.Adam(learning_rate=lrvec[i]), loss="mse", metrics="mse"
    )
    mdl.fit(
        x=t,
        y=[x, f],
        epochs=epvec[i],
        batch_size=N_train,
        verbose=0,
        callbacks=[tqdm_callback],
    )
    # mdl.fit(x=t, y=[x, f], epochs=epvec[i], batch_size=N_train)

xh, fh = mdl.predict(t)

plt.figure(figsize=[14, 5])
plt.subplot(121)
plt.plot(t, x, "b-", t, xh, "ro")
plt.legend(["y", "xh"])

plt.subplot(122)
plt.plot(t, f, "b-", t, fh, "ro")
plt.legend(["f", "fh"])
plt.show()

print(
    "Values identified for the ODE (rho = 2.0, b = 0.084, k = 2.2020 are the real values)"
)
print(mdl.weights[0])
print(mdl.weights[1])

mdl.summary()

a = 1
