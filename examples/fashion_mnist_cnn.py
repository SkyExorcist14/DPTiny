
import random
import time
import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)
from dptiny.data import DataLoader
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam
from dptiny.data.fashion_mnist import get_fashion_mnist, CLASSES


# ---------------------------------------------------------
# Random seed
# ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")


# ---------------------------------------------------------
# Load Fashion-MNIST
# ---------------------------------------------------------
print("Loading Fashion-MNIST dataset...")

X_train, X_test, y_train, y_test = get_fashion_mnist(
    normalize=False,
    flatten=False,
)


# ---------------------------------------------------------
# Standardize using training mean and standard deviation
# ---------------------------------------------------------
mean = X_train.mean()
std = X_train.std()

print(f"Training mean: {mean:.6f}")
print(f"Training std:  {std:.6f}")

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ---------------------------------------------------------
# Move data to GPU
# ---------------------------------------------------------
if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)


# ---------------------------------------------------------
# CNN model
# Same architecture as examples/mnist_cnn.py
# ---------------------------------------------------------
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)

if is_gpu():
    model.to_gpu()


# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------
batch_size = 64
max_epoch = 10

data_loader = DataLoader(
    (X_train, y_train),
    batch_size,
)

test_loader = DataLoader(
    (X_test, y_test),
    batch_size,
    shuffle=False,
)

optimizer = Adam(model, lr=0.001)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
start_time = time.time()

for epoch in range(max_epoch):

    sum_loss = 0.0
    sum_correct = 0
    count = 0

    model.train()

    for x, t in data_loader:

        x = Variable(x)

        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        pred = y.data.argmax(axis=1)

        sum_loss += float(loss.data) * len(t)
        sum_correct += int((pred == t).sum())
        count += len(t)

    train_loss = sum_loss / count
    train_acc = sum_correct / count


    # -----------------------------------------------------
    # Test accuracy
    # -----------------------------------------------------
    test_correct = 0
    test_count = 0

    model.eval()

    with test_mode(), no_grad():

        for x, t in test_loader:

            y = model(Variable(x))
            pred = y.data.argmax(axis=1)

            test_correct += int((pred == t).sum())
            test_count += len(t)

    test_acc = test_correct / test_count


    print(
        f"Epoch {epoch + 1:2d}/{max_epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )


# ---------------------------------------------------------
# Final evaluation and confusion matrix
# ---------------------------------------------------------
confusion_matrix = np.zeros((10, 10), dtype=np.int64)

model.eval()

with test_mode(), no_grad():

    for x, t in test_loader:

        y = model(Variable(x))
        pred = y.data.argmax(axis=1)

        for true_label, predicted_label in zip(t, pred):
            confusion_matrix[
                int(true_label),
                int(predicted_label)
            ] += 1


# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix)


print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(CLASSES):

    total = confusion_matrix[i].sum()
    correct = confusion_matrix[i, i]
    accuracy = correct / total

    print(
        f"{i}: {class_name:12s} "
        f"{accuracy:.4f}"
    )


# ---------------------------------------------------------
# Most confused pair of classes
# ---------------------------------------------------------
pair_scores = confusion_matrix + confusion_matrix.T
np.fill_diagonal(pair_scores, 0)

i, j = np.unravel_index(
    np.argmax(pair_scores),
    pair_scores.shape
)

print("\n" + "=" * 60)
print("MOST CONFUSED CLASS PAIR")
print("=" * 60)

print(
    f"{CLASSES[i]} -> {CLASSES[j]}: "
    f"{confusion_matrix[i, j]} misclassified samples"
)

print(
    f"{CLASSES[j]} -> {CLASSES[i]}: "
    f"{confusion_matrix[j, i]} misclassified samples"
)

print(
    f"\nTraining completed in "
    f"{time.time() - start_time:.2f} seconds"
)
