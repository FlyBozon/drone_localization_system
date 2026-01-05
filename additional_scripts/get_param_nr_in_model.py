#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf

#import h5py for HDF5 fallback
try:
    import h5py
except Exception:
    h5py = None

from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def swish(x):
    return tf.nn.swish(x)

@register_keras_serializable(package="Custom")
def relu6(x):
    return tf.nn.relu6(x)

@register_keras_serializable(package="Custom")
class FixedDropout(tf.keras.layers.Dropout):
    # efficientnet.model.FixedDropout often used; match typical signature
    def __init__(self, rate, seed=None, noise_shape=None, **kwargs):
        super().__init__(rate=rate, seed=seed, noise_shape=noise_shape, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# register aliases in get_custom_objects too (helps older models)
try:
    tf.keras.utils.get_custom_objects().update({
        "swish": swish,
        "relu6": relu6,
        "FixedDropout": FixedDropout,
    })
except Exception:
    pass

def count_params_from_model(model):
    # robust counting for TF2 / Keras3
    trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    return trainable, non_trainable, trainable + non_trainable

def count_params_from_h5(path):
    if h5py is None:
        raise RuntimeError("h5py not installed; cannot read HDF5 fallback.")
    total = 0
    def visitor(name, obj):
        nonlocal total
        # datasets in HDF5 that represent weights normally are numeric datasets
        if isinstance(obj, h5py.Dataset):
            # ignore empty shapes
            shape = obj.shape
            if len(shape) == 0:
                return
            total += int(np.prod(shape))
    try:
        with h5py.File(path, "r") as f:
            f.visititems(visitor)
    except Exception as e:
        raise
    return int(total)

def is_hdf5_file(path):
    # quick check: try opening with h5py
    if h5py is None:
        return False
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def analyze_models(folder):
    #print("\n=== MODEL ANALYSIS ===\n")

    files = sorted(os.listdir(folder))
    for fname in files:
        path = os.path.join(folder, fname)

        # only consider files with common model extensions or directories (SavedModel)
        if not (fname.endswith(".keras") or fname.endswith(".h5") or fname.endswith(".hdf5") or os.path.isdir(path) or fname.endswith(".pb")):
            continue

        size_mb = os.path.getsize(path) / (1024**2) if os.path.isfile(path) else None

        print(f"\n Model: {fname}")
        print(f"   Path: {path}")
        if size_mb is not None:
            print(f"   File Size: {size_mb:.2f} MB")

        # Attempt 1: try to load model normally (with custom objects registered above)
        try:
            # compile=False to speed up and avoid custom losses/metrics issues
            model = tf.keras.models.load_model(path, compile=False)
            trainable, non_trainable, total = count_params_from_model(model)
            print(f"    Loaded with tf.keras.models.load_model")
            print(f"    Trainable params:     {trainable:,}")
            print(f"    Non-trainable params: {non_trainable:,}")
            print(f"    Total params:         {total:,}")
            continue
        except Exception as e:
            # show brief cause (first line) and continue to fallback
            err_msg = str(e).splitlines()[0]
            print("    Cannot load model (load_model):")
            print(f"      {err_msg}")

        # Attempt 2: HDF5 fallback (for .h5 or HDF5 .keras)
        try:
            if os.path.isfile(path) and is_hdf5_file(path):
                total = count_params_from_h5(path)
                print("    HDF5 fallback succeeded (counted params from file contents).")
                print(f"    Total params (approx): {total:,}")
                continue
        except Exception as e:
            print("    HDF5 fallback failed:")
            print(f"      {e}")

        # Attempt 3: try TensorFlow checkpoint listing (for SavedModel directories)
        try:
            if os.path.isdir(path):
                # look for variables/ checkpoint inside SavedModel dir
                variables_dir = os.path.join(path, "variables")
                if os.path.isdir(variables_dir):
                    # try to find checkpoint prefix
                    # tf.train.list_variables expects checkpoint prefix (e.g. path/variables/variables)
                    ckpt_prefix = os.path.join(variables_dir, "variables")
                    try:
                        var_list = tf.train.list_variables(ckpt_prefix)
                        total = 0
                        for name, shape in var_list:
                            total += int(np.prod(shape))
                        print("    SavedModel variables fallback succeeded.")
                        print(f"    Total params (approx): {total:,}")
                        continue
                    except Exception as e:
                        print("    SavedModel checkpoint read failed:")
                        print(f"      {e}")
        except Exception:
            pass
    print("\nDONE\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "trained_models"
    if not os.path.exists(folder):
        #print(f"Folder not found: {folder}")
        sys.exit(1)
    analyze_models(folder)
