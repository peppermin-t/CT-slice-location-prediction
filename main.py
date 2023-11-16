import numpy as np
import matplotlib.pyplot as plt


# Retrieve pre-processed data
data = np.load('ct_data.npz')
X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']


# Verify the length of training and validation set
print(len(y_train))
print(len(y_val))

# Verify that the mean of training positions is 0
print(np.allclose(np.mean(y_train), 0))
# Verify that the mean of validation positions is not 0
print(np.allclose(np.mean(y_val), 0))
# the standard error of the validation positions
print(np.mean(y_val), np.std(y_val) / np.sqrt(len(y_val)))
# the standard error of the first 5785 entries of training positions
y_part = y_train[: 5785]
print(np.mean(y_part), np.std(y_part) / np.sqrt(len(y_part)))


# Identify the input features with constant values and discard them
ft_const = np.all(X_train == X_train[0, ], axis=0)
# Update the input data matricies
X_train = X_train[: , ~ ft_const]; X_val = X_val[: , ~ ft_const]; X_test = X_test[: , ~ ft_const]

# Identify the dupicated input features and keep only the first
X_train_, ft_unique = np.unique(X_train, axis=1, return_index=True)
# Calculate the dicarded indices in this satge
ft_dup = np.setdiff1d(np.arange(X_train.shape[1]), ft_unique)
# Update the input data matricies
X_train = X_train[: , ft_unique]; X_val = X_val[: , ft_unique]; X_test = X_test[: , ft_unique]

# Report the dicarded columns (with stage-wise indices)
print(np.where(ft_const))
print(ft_dup)


def fit_linreg(X, yy, alpha):
	pass
