Physics-informed Artificial Neural Network

Author: Helon Ayala, October 2020

Based on: https://github.com/maziarraissi/DeepVIV

(translated to Keras layer subclassing and tf 2.0)

# Quick description

In this notebook we reproduce the results of the paper 

* Raissi, Maziar, Zhicheng Wang, Michael S. Triantafyllou, and George Em Karniadakis."Deep learning of vortex-induced vibrations." Journal of Fluid Mechanics 861 (2019): 119-137.

in particular Section 2.1. Please refer to the paper for a detailed description of the method and case study. In the following we restrict to the information needed for the notebook to be self-contained.

# Pending

Extend it to x,y,t independent variables (for e.g. NS-eqs.). Step-by-step:
* add more outputs to  `class tToXandF(tf.keras.layers.Layer):` , specifically in the line `self.denseList.append(tf.keras.layers.Dense(1))  # output layer`
* add more gradients to the environment `with tf.GradientTape(`
* change input-output data and loss function
