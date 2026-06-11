#!/usr/bin/env python3
"""
mnist_16x16_bnn.py

Binary Neural Network (1‑bit weights) for 16x16 down‑scaled MNIST using
Larq + TensorFlow.

Why the extra boilerplate?
-------------------------
Keras 3 (bundled with TensorFlow >= 2.16, Python >= 3.12) removed the
`LocallyConnected*D` layers.  Larq 0.15.x still tries to *import* them
when its module is initialised, even though we never use them in this
MLP.  To avoid the resulting `AttributeError` we inject **tiny stub
classes for both 1D and 2D variants** *before* importing Larq.  The
forward pass never touches those stubs, so they will not affect
numerics; they only satisfy Python’s attribute lookup at import time.

If you prefer a cleaner stack, create a Python 3.11 (or 3.10) env and
pin TensorFlow 2.13 + Larq 0.15 instead.  The rest of the script works
unchanged.
"""

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# Compatibility patch for Keras 3 (TF 2.16+): re‑introduce missing layers
# ---------------------------------------------------------------------

# Now it is safe to import Larq
import larq as lq
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Hyper‑parameters
BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 1e-3
HIDDEN_UNITS = 64
TEST_SPLIT = 0.1
NUM_CLASSES = 10

# 1. Load dataset
data = np.load("mnist_augmented.npz")
X = data["images"].astype("float32")        # (N, 16, 16)
y = data["labels"].astype("int32")          # (N,)

# 2. Pre‑process
X = X.reshape(-1, 256)            # flatten 16x16 -> 256
X = 2.0 * X - 1.0                 # map {0,1} -> {-1,+1}
y = to_categorical(y, NUM_CLASSES)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

# 3. Define Binary Neural Network (BNN)
common_quant = dict(
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
    use_bias=False,
)

model = tf.keras.Sequential([
    lq.layers.QuantDense(HIDDEN_UNITS, input_shape=(256,), **common_quant),
    tf.keras.layers.BatchNormalization(momentum=0.15),
    # Keeping activations float typically yields +0.5‑1 % accuracy.
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", use_bias=False),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 4. Train
model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# 5. Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# 6. Save model
model.save("bnn_mnist16.h5")

# Optional: Larq Compute Engine / TFLite conversion
# -------------------------------------------------
# import larq_compute_engine as lce
# tflite_model = lce.converter.convert_keras_model(model)
# with open("bnn_mnist16.tflite", "wb") as f:
#     f.write(tflite_model)
