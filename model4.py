#!/usr/bin/env python3
"""
mnist_16x16_bnn.py  — **all‑in‑one**
===================================
1.  Train a small Binary Neural Network (Larq + TensorFlow)
2.  Export every weight & BN statistic to a `.npz`
3.  Provide a **pure‑NumPy** runtime (`predict_numpy`) so you can deploy
    the network on any CPU without Keras/TensorFlow/Larq.

Run modes
---------
```bash
# A. Full training (default) ➜ exports weights after training
python mnist_16x16_bnn.py train

# B. Skip training, just (re‑)export weights from an existing HDF5 / Keras file
python mnist_16x16_bnn.py export bnn_mnist16.h5

# C. CPU‑only inference demo on 16×16 binary MNIST images
python mnist_16x16_bnn.py infer mnist_augmented.npz bnn_weights.npz
```

Dependencies for **training**: `tensorflow`, `tf_keras`, `larq`,
`scikit‑learn`.  For **inference only** you just need `numpy`.

Tested with:
* Python 3.12 + TensorFlow 2.17 + Larq 0.15.6
* Also works on Python 3.10 / TF 2.13 (set `TF_USE_LEGACY_KERAS=1` only
  for TF ≥ 2.16).
"""

import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np

# ---------------------------------------------------------------------
# Optional framework imports (only needed for training / export)
# ---------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "train"

if MODE in {"train", "export"}:
    # Ensure Larq can import under TF 2.16+/Keras 3
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    import tensorflow as tf

    # Patch back the missing LocallyConnected layers so Larq ≥0.15 imports
    for _name, _parent in {
        "LocallyConnected1D": tf.keras.layers.Conv1D,
        "LocallyConnected2D": tf.keras.layers.Conv2D,
    }.items():
        if not hasattr(tf.keras.layers, _name):
            type(_name, (_parent,), {})
            setattr(tf.keras.layers, _name, globals()[_name])

    import larq as lq
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------
# Hyper‑parameters
# ---------------------------------------------------------------------
BATCH_SIZE = 512
EPOCHS = 1000
LEARNING_RATE = 1e-3
HIDDEN_UNITS = 64
TEST_SPLIT = 0.1
NUM_CLASSES = 10
WEIGHT_FILE = "bnn_weights.npz"
MODEL_FILE = "bnn_mnist16.keras"

# ---------------------------------------------------------------------
# Utility: export weights → npz
# ---------------------------------------------------------------------

def export_weights_npz(model, out_path: str = WEIGHT_FILE) -> None:
    """Save all trainable weights + BN moving stats in NumPy format."""
    weights: Dict[str, np.ndarray] = {}

    for layer in model.layers:
        if isinstance(layer, (lq.layers.QuantDense, tf.keras.layers.Dense)):
            w = layer.get_weights()[0]
            key = f"{layer.name}_W"
            # First layer is binary (–1/+1); compress as int8 to halve size
            if isinstance(layer, lq.layers.QuantDense):
                w = w.astype(np.int8)
            weights[key] = w

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            prefix = layer.name
            weights[f"{prefix}_gamma"] = gamma.astype(np.float32)
            weights[f"{prefix}_beta"] = beta.astype(np.float32)
            weights[f"{prefix}_mean"] = mean.astype(np.float32)
            weights[f"{prefix}_var"] = var.astype(np.float32)

    np.savez_compressed(out_path, **weights)
    print(f"[✓] Weights written → {out_path} ({Path(out_path).stat().st_size / 1024:.1f} KB)")

# ---------------------------------------------------------------------
# Pure‑NumPy inference implementation
# ---------------------------------------------------------------------

def load_numpy_weights(path: str = WEIGHT_FILE) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path, allow_pickle=False).items()}


def predict_numpy(batch16x16: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised forward pass for a batch of 16×16 binary images.

    * batch16x16: shape (N, 16, 16) with values 0/1
    * returns: probabilities shape (N, 10)
    """
    # Flatten and map to –1/+1
    x = batch16x16.reshape(batch16x16.shape[0], 256).astype(np.float32)
    x = 2.0 * x - 1.0

    # --- First binary dense ------------------------------------------
    W0 = w["quant_dense_W"].astype(np.float32)  # (256, 64)
    z0 = np.matmul(x, W0)  # (N, 64)

    # BatchNorm0
    g0, b0, m0, v0 = (w[k] for k in (
        "batch_normalization_gamma",
        "batch_normalization_beta",
        "batch_normalization_mean",
        "batch_normalization_var",
    ))
    eps = 1e-3
    bn0 = g0 * (z0 - m0) / np.sqrt(v0 + eps) + b0

    # ReLU
    a0 = np.maximum(bn0, 0.0)

    # --- Final dense (float) -----------------------------------------
    W1 = w["dense_W"].astype(np.float32)  # (64, 10)
    logits = np.matmul(a0, W1)  # (N, 10)

    # Softmax
    logits -= logits.max(axis=1, keepdims=True)  # for numerical stability
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs

# ---------------------------------------------------------------------
# Train‑and‑export pipeline (MODE == 'train')
# ---------------------------------------------------------------------

if MODE == "train":
    print("[TRAIN] Loading dataset …")
    data = np.load("mnist_augmented.npz")
    X = data["images"].astype("float32")
    y = data["labels"].astype("int32")

    # Pre‑process
    X = X.reshape(-1, 256) * 2.0 - 1.0  # flatten + rescale to ±1
    y_cat = to_categorical(y, NUM_CLASSES)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_cat, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    # Build BNN
    quant_cfg = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )

    model = tf.keras.Sequential([
        lq.layers.QuantDense(HIDDEN_UNITS, input_shape=(256,), name="quant_dense", **quant_cfg),
        tf.keras.layers.BatchNormalization(name="batch_normalization"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", use_bias=False, name="dense"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[TRAIN] Validation accuracy: {val_acc:.4f}")

    # Save Keras model + raw weights
    model.save(MODEL_FILE)
    export_weights_npz(model, WEIGHT_FILE)

# ---------------------------------------------------------------------
# Export‑only mode: read .keras/.h5 and dump weights
# ---------------------------------------------------------------------

elif MODE == "export":
    mdl_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_FILE
    print(f"[EXPORT] Loading model from {mdl_path} …")
    import tensorflow as tf  # still need TF to load the model
    model = tf.keras.models.load_model(mdl_path, compile=False)
    export_weights_npz(model, WEIGHT_FILE)

# ---------------------------------------------------------------------
# Pure‑NumPy inference demo (MODE == 'infer')
# ---------------------------------------------------------------------

elif MODE == "infer":
    if len(sys.argv) < 4:
        print("Usage: python mnist_16x16_bnn.py infer <dataset.npz> <weights.npz>")
        sys.exit(1)
    data_path, w_path = sys.argv[2:4]
    print(f"[INFER] Loading weights from {w_path}")
    weights = load_numpy_weights(w_path)

    print(f"[INFER] Loading dataset from {data_path}")
    with np.load(data_path) as d:
        images = d["images"].astype(np.float32)
        labels = d["labels"].astype(np.int32)

    probs = predict_numpy(images, weights)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean()
    print(f"Accuracy (NumPy runtime): {acc:.4f}")

else:
    print(f"Unknown MODE '{MODE}'. Use train | export | infer")
