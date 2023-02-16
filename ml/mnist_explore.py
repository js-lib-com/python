import pickle

description = """
MNIST pickle file contains three tuples, for training, validation respective testing sets.
All three tuples have the same structure:
- a table (two dimensions array) for images,
- an array of labels.

Both images table and labels array have the same length. An image from images table is an
array of 784 float values, normalized in [0.0 .. 1.0] range, representing gray scale. 
"""
print(description)

with open("mnist.pkl", "rb") as file:
    # MNIST pickle file contains three tuples, for training, validation respective testing sets
    # pickle.load will initialize all tree variables with respective tuples

    # if use a single variable (instead of three) pickle.load will return a single (tuple) value
    # data_set = pickle.load(file, encoding="latin1")
    # train_set = data_set[0]
    # valid_set = data_set[1]
    # test_set = data_set[2]

    # it is critical to use encoding="latin1"

    train_set, valid_set, test_set = pickle.load(file, encoding="latin1")
    print(f"loaded {file.tell()} bytes from MNIST file {file.name}")

print()
print("train set dump:")
print(train_set)

train_images = train_set[0]
train_labels = train_set[1]
print()
print(f"train set images shape: {train_images.shape}")
print(f"train set labels shape: {train_labels.shape}")

print()
print("valid set dump:")
print(valid_set)

valid_images = valid_set[0]
valid_labels = valid_set[1]
print()
print(f"valid set images shape: {valid_images.shape}")
print(f"valid set labels shape: {valid_labels.shape}")

print()
print("test set dump:")
print(test_set)

test_images = test_set[0]
test_labels = test_set[1]
print()
print(f"test set images shape: {test_images.shape}")
print(f"test set labels shape: {test_labels.shape}")
