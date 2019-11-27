
import crypten
import torch
from examples.mpc_linear_svm.mpc_linear_svm import train_linear_svm, evaluate_linear_svm
from crypten import mpc
import crypten.communicator as comm

ALICE = 0
BOB = 1
crypten.init()
num_features = 100
num_train_examples = 1000
num_test_examples = 100
epochs = 40
lr = 3.0

# Set random seed for reproducibility
torch.manual_seed(1)

features = torch.randn(num_features, num_train_examples)
w_true = torch.randn(1, num_features)
b_true = torch.randn(1)

labels = w_true.matmul(features).add(b_true).sign()

test_features = torch.randn(num_features, num_test_examples)
test_labels = w_true.matmul(test_features).add(b_true).sign()

# Specify file locations to save each piece of data
filenames = {
    "features": "/tmp/features.pth",
    "labels": "/tmp/labels.pth",
    "features_alice": "/tmp/features_alice.pth",
    "features_bob": "/tmp/features_bob.pth",
    "samples_alice": "/tmp/samples_alice.pth",
    "samples_bob": "/tmp/samples_bob.pth",
    "w_true": "/tmp/w_true.pth",
    "b_true": "/tmp/b_true.pth",
    "test_features": "/tmp/test_features.pth",
    "test_labels": "/tmp/test_labels.pth",
}


def save_all_data():
    # Save features, labels for Data Labeling example
    crypten.save(features, filenames["features"])
    crypten.save(labels, filenames["labels"])

    # Save split features for Feature Aggregation example
    features_alice = features[:50]
    features_bob = features[50:]

    print(comm.get().get_world_size())
    print(comm.get())



    # Save split dataset for Dataset Aggregation example
    samples_alice = features[:, :500]
    samples_bob = features[:, 500:]


    crypten.save(features_alice, filenames["features_alice"], ALICE)
    crypten.save(samples_alice, filenames["samples_alice"], ALICE)
    # Save true model weights and biases for Model Hiding example
    crypten.save(w_true, filenames["w_true"], ALICE)
    crypten.save(b_true, filenames["b_true"], ALICE)

    # crypten.save(features_bob, filenames["features_bob"], BOB)
    # crypten.save(samples_bob, filenames["samples_bob"], BOB)
    # crypten.save(test_features, filenames["test_features"], BOB)
    # crypten.save(test_labels, filenames["test_labels"], BOB)

    # Alice loads some features, Bob loads other features
    features_alice_enc = crypten.load(filenames["features_alice"], src=ALICE)
    print(features_alice_enc)
    features_bob_enc = crypten.cryptensor(features_bob, ptype=crypten.arithmetic)
    # Concatenate features
    features_enc = crypten.cat([features_alice_enc, features_bob_enc], dim=0)

    # Encrypt labels
    labels_enc = crypten.cryptensor(labels)

    # Execute training
    w, b = train_linear_svm(features_enc, labels_enc, epochs=epochs, lr=lr)

    # Evaluate model
    evaluate_linear_svm(test_features, test_labels, w, b)

def mpc_test():
    # Constructing CrypTensors with ptype attribute

    x = crypten.cryptensor([1.0, 2.0, 3.0])
    print("x:", x)
    print("x get_plain_text:", x.get_plain_text())

    # arithmetic secret-shared tensors
    x_enc = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.arithmetic)
    print("x_enc internal type:", x_enc.ptype)

    # binary secret-shared tensors
    y = torch.tensor([1, 2, 1], dtype=torch.int32)
    y_enc = crypten.cryptensor(y, ptype=crypten.binary)
    print("y_enc internal type:", y_enc.ptype)
    x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.arithmetic)

    rank = comm.get().get_rank()
    print(f"Rank {rank}:\n {x_enc}")
    print(f"Rank {rank}:\n {x_enc.get_plain_text()}")

def initm():
    save_all_data()




if __name__ == "__main__":
    initm()