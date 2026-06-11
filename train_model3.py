import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import larq as lq
from sklearn.model_selection import train_test_split

# ── 1. Load and prepare the augmented data ──────────────────────────────────────
data = np.load("mnist_augmented4.npz")          # images: (-1, 16, 16) uint8
images = data["images"].astype("float32") / 255.0   # 0 or 1
images = 2.0 * images - 1.0                     # map {0,1} -> {-1,+1}
labels = data["labels"].astype("int64")

images = images.reshape((-1, 16, 16, 1))       # → (-1, 16, 16, 1)

# stratified 90 / 10 split keeps class balance
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels
)

# data_train = np.load("mnist_augmented.npz")
# data_test = np.load("mnist_augmented4.npz")
# X_train = data_train["images"].astype("float32") / 255.0
# X_train = 2.0 * X_train - 1.0
# y_train = data_train["labels"].astype("int64")
# X_test = data_test["images"].astype("float32") / 255.0
# X_test = 2.0 * X_test - 1.0
# y_test = data_test["labels"].astype("int64")
# X_train = X_train.reshape((-1, 16, 16, 1))       # → (240000, 16, 16, 1)
# X_test = X_test.reshape((-1, 16, 16, 1))         # → (60000, 16, 16, 1)

# ── 2. Build a 16×16 binary network ────────────────────────────────────────────
bin_kwargs = dict(
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
)

model = tf.keras.Sequential([
    lq.layers.QuantConv2D(32, (5, 5), use_bias=False,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        input_shape=(16, 16, 1)
    ),

    tf.keras.layers.Flatten(),

    lq.layers.QuantDense(10, use_bias=False,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    ),
    tf.keras.layers.Activation("softmax"),
])

lq.models.summary(model)

# ── 3. Compile & train ─────────────────────────────────────────────────────────
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=2048, epochs=240, validation_split=0.1)

# ---------- Get predictions on the full test set ----------
# (quantized_scope not strictly necessary for inference, but harmless)
with lq.context.quantized_scope(True):
    y_prob = model.predict(X_test, batch_size=128, verbose=0)
    weights = model.get_weights()

# save weights to file
np.savez_compressed("weights.npz", *weights)

y_pred = np.argmax(y_prob, axis=1)

mis_idx  = np.where(y_pred != y_test)[0]                       # wrong samples
n_errors = len(mis_idx)

# convert images back to 0‑255 uint8 and flatten to 1×256
imgs_uint8 = (((X_test[mis_idx] + 1.0) / 2.0) * 255).astype("uint8")
imgs_flat  = imgs_uint8.reshape(n_errors, -1)                  # (N, 256)

# build final matrix: index | truth | pred | 256 pixels
meta       = np.column_stack((mis_idx, y_test[mis_idx], y_pred[mis_idx]))
out        = np.hstack((meta, imgs_flat))

header = (
    "index,true_label,predicted_label,"
    + ",".join(f"p{i}" for i in range(256))                   # p0 … p255
)

np.savetxt(
    "misclassified_test_samples.csv",
    out,
    fmt="%d",
    delimiter=",",
    header=header,
    comments="",
)

print(f"Accuracy: {100 * (1 - n_errors/len(y_test)):.2f}%")
print(f"Saved {n_errors} mis‑classifications (with images) to misclassified_test_samples.csv")