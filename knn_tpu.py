import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

def compute_diff(u, v):
    return (u[:, None] - v[None, :]) ** 2

compute_diff = jax.vmap(compute_diff, in_axes=1, out_axes=-1)

# We pmap the built-in argsort function along the first axes.
# We sort a matrix with shape (devices, n_test // devices, n_train)
p_argsort = jax.pmap(jnp.argsort, in_axes=0)

def compute_distance(U, V):
    return compute_diff(U, V).mean(axis=-1)

n_train = 30_000
n_test = 10 * 8

devices = 8
def compute_k_closest(U, V, k):
    D = compute_distance(U, V)
    D = D.reshape(devices, n_test // devices, -1)
    nearest = p_argsort(D)[..., 1:k+1]
    return nearest

train, test = get_datasets()

X_train = train["image"].reshape(-1, 28 ** 2)
X_train = 4 * X_train - 2
y_train = train["label"]

X_test = test["image"].reshape(-1, 28 ** 2)
X_test = 4 * X_test - 2
y_test = test["label"]

k = 20
train, test = get_datasets()

X_train = train["image"].reshape(-1, 28 ** 2)
X_train = 4 * X_train - 2
y_train = train["label"]

X_test = test["image"].reshape(-1, 28 ** 2)
X_test = 4 * X_test - 2
y_test = test["label"]

X_train = X_train[:n_train]
y_train = y_train[:n_train]

X_test = X_test[:n_test]
y_test = y_test[:n_test]


k_nearest = compute_k_closest(X_test, X_train, k)
if jax.process_index() == 0:
    print(k_nearest.shape)
    k_nearest = k_nearest.reshape(-1, k)
    class_rate = (y_train[k_nearest, ...].mean(axis=1).round() == y_test).mean()
    print(f"{class_rate=}")