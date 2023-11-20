import numpy as np
import matplotlib.pyplot as plt


# Retrieve pre-processed data
data = np.load('ct_data.npz')
X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']


# print("####################")
# print("Question 1a:")
# # Verify the length of training and validation set
# print(len(y_train))
# print(len(y_val))

# # Verify that the mean of training positions is 0
# print(np.allclose(np.mean(y_train), 0))
# # Verify that the mean of validation positions is not 0
# print(np.allclose(np.mean(y_val), 0))
# # the standard error of the validation positions
# print(np.mean(y_val), np.std(y_val) / np.sqrt(len(y_val)))
# # the standard error of the first 5785 entries of training positions
# y_part = y_train[: 5785]
# print(np.mean(y_part), np.std(y_part) / np.sqrt(len(y_part)))


print("####################")
print("Question 1b:")
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


print("####################")
print("Question 2:")
def fit_linreg(X, yy, alpha=30):
    '''
    fit a regularized linear regression model with data augmentation and lstsq solver.
     Use data augnentation approach and solve y' ~ X'w with a lstsq solver,
     minimizing the regularized least squares cost (bias not regularized):

       - np.sum(-yy*(np.dot(X, ww) + bb)) + alpha*np.dot(ww,ww)

     Inputs:
             X N,D design matrix of input features
            yy N,  real-valued targets
         alpha     scalar regularization constant

     Outputs:
            ww D,  fitted weights
            bb     scalar fitted bias
    '''
    D = X.shape[0]
    # Construct the design matrix
    phi = np.hstack([np.ones((D, 1)), X])
    # Construct the augmenting diagnal matrix
    diag_aug = np.sqrt(alpha) * np.identity(phi.shape[1])
    # not regularizing bias, make the position corresponding to bias 0
    diag_aug[0, 0] = 0
    # Concatenate the augmenting matrix to the original design matrix vertically
    phi_reg = np.vstack([phi, diag_aug])
    # Concatenate a 0 vector to the target vector
    y_reg = np.hstack([yy, np.zeros(phi.shape[1])])

    # generate the fitted weight with lstsq
    w_fit = np.linalg.lstsq(phi_reg, y_reg, rcond=None)[0]

    # separately returning weights and bias
    ww, bb = w_fit[1: ], w_fit[0]

    return ww, bb

from support_code import *

def get_RMSE_lin(ww, bb, X, yy):
    '''
    get the root-mean-square-loss (RMSE) of given data for a linear fit 

     RMSE = np.sqrt(np.mean((yy - yy_fit) ** 2))
     Inputs:
             ww D, fitted weights
             bb    fitted bias
             X N,D design matrix of input features
            yy N,  real-valued targets

     Outputs:
            RMSE   RMSE of given data for the linear fit 
    '''
    # calculate the predicted targets under the fitted values
    yy_fit = X.dot(ww) + bb
    # return the RMSE of given data for the linear fit
    return np.sqrt(np.mean((yy - yy_fit) ** 2))

# Specific the regularization constant alpha as 30
alpha = 30
# Fit the linear model using self-implemented fit_linreg
ww_1, bb_1 = fit_linreg(X_train, y_train, alpha)
# Fit the linear model using provided fit_linreg_gradopt
ww_2, bb_2 = fit_linreg_gradopt(X_train, y_train, alpha)
# Report the RMSE on the training set for parameters fitted with fit_linreg
print(get_RMSE_lin(ww_1, bb_1, X_train, y_train))
# Report the RMSE on the training set for parameters fitted with fit_linreg_gradopt
print(get_RMSE_lin(ww_2, bb_2, X_train, y_train))
# Report the RMSE on the validation set for parameters fitted with fit_linreg
print(get_RMSE_lin(ww_1, bb_1, X_val, y_val))
# Report the RMSE on the validation set for parameters fitted with fit_linreg_gradopt
print(get_RMSE_lin(ww_2, bb_2, X_val, y_val))


print("####################")
print("Question 3:")

def fit_logreg_gradopt(X, yy, alpha):
    """
    fit a regularized logistic regression model with gradient opt

         ww, bb = fit_logreg_gradopt(X, yy, alpha)

     Find weights and bias by using a gradient-based optimizer
     (minimize_list) to improve the regularized negative log likelihood
       least squares cost:

       - np.sum(np.log(1 / (1 + np.exp(-yy*(np.dot(X, ww) + bb))))) + alpha*np.dot(ww,ww)

     Inputs:
             X N,D design matrix of input features
            yy N,  real-valued targets
         alpha     scalar regularization constant

     Outputs:
            ww D,  fitted weights
            bb     scalar fitted bias
    """
    D = X.shape[1]
    # create the arguments needed for calculating costs in minimize
    args = (X, yy, alpha)
    # generate the initializing values for ww and bb
    init = (np.zeros(D), np.array(0))
    # fit the model with minimize_list, using cost function logreg_cost
    # functions imported in Q2
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

# number of thresholded classification problems to fit
K = 20
# K = 40
# Caculate the range of y_train values and the step size w.r.t K thresholds
mx = np.max(y_train); mn = np.min(y_train); hh = (mx-mn)/(K+1)
# Create the threshold ndarray w.r.t. number K 
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)

## logistic regression fit
# Initialize the ndarray to store the logistic regression result w.r.t each class
prob_train = np.zeros((len(y_train), K))
prob_val = np.zeros((len(y_val), K))

# Initialize the logistic regression parameters for K classes
ww_V = np.zeros((K, X_train.shape[1]))
bb_k = np.zeros(K)

# Loop over K logistical regression problems
for kk in range(K):
    # Generate the new labels under logistic regression settings
    labels = y_train > thresholds[kk]
    # fit logistic regression to these labels (using alpha from Q2)
    ww_V[kk, :], bb_k[kk] = fit_logreg_gradopt(X_train, labels, alpha)
    # Generate the predicted probabilities for this class with fitted parameters
    prob_train[:, kk] = 1 / (1 + np.exp(- (np.dot(X_train, ww_V[kk, :]) + bb_k[kk])))
    prob_val[:, kk] = 1 / (1 + np.exp(- (np.dot(X_val, ww_V[kk, :]) + bb_k[kk])))

# linear regression fit to the probabilities for each logistic regression and the final label
# (fit_linreg_gradopt imported in Q2)
ww_lin, bb_lin = fit_linreg_gradopt(prob_train, y_train, alpha)
# calculate RMSE of the training set and validation set with get_RMSE_lin (from Q2)
print(get_RMSE_lin(ww_lin, bb_lin, prob_train, y_train))
print(get_RMSE_lin(ww_lin, bb_lin, prob_val, y_val))


print("####################")
print("Question 4:")
# Set seed to get consistent results for comparison
np.random.seed(10)
def fit_nn(X, yy, alpha, init):
    """
    fit a regularized neural network model with gradient opt
     Find parameters by using a gradient-based optimizer
     (minimize_list) to minimize the regularized least squares cost:

       np.dot(np.dot(1 / (1 + np.exp(-np.dot(X, V.T) + bk)), ww) + bb - yy) 
        + alpha*(np.sum(V*V) + np.dot(ww,ww))

     Inputs:
             X N,D input design matrix
            yy N,  regression targets
         alpha     scalar regularization for weights

     Outputs:
            ww K,  hidden-output weights
            bb     scalar output bias
            V K,D hidden-input weights
            bk K,  hidden biases
    """
    # Create the arguments needed for calculating costs in minimize
    args = (X, yy, alpha)
    # fit the model with minimize_list, using cost function nn_cost
    # functions imported in Q2
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return ww, bb, V, bk

def get_RMSE_nn(params, X, yy):
    '''
    get the root-mean-square-loss (RMSE) of given data for a neural network

     RMSE = np.sqrt(np.mean((yy - yy_fit) ** 2))
     Inputs:
             ww D,  fitted weights
             bb     fitted bias
              V K,D hidden-input weights
             bk K,  hidden biases
              X N,D design matrix of input features
             yy N,  real-valued targets

     Outputs:
            RMSE   RMSE of given data for the nn
    '''
    # calculate the predicted targets under the fitted values
    # (nn_cost imported in Q2)
    nn_fit = nn_cost(params, X)
    # return the RMSE of given data for the neural network
    return np.sqrt(np.mean((yy - nn_fit) ** 2))

D = X_train.shape[1]
# Create a sensible random initialization of the parameters (uniform(-1, 1) here)
uni_init = (np.random.uniform(-1, 1, K), np.random.uniform(-1, 1),
            np.random.uniform(-1, 1, (K, D)), np.random.uniform(-1, 1, K))
# Retrieve the parameters pretrained in logistic regressions from Q3 for initialization
pretrain_init = (ww_lin, bb_lin, ww_V, bb_k)

# nn fit with sensible random initialization
params_nn_a = fit_nn(X_train, y_train, alpha, init=uni_init)
# Report the RMSE of training set
print(get_RMSE_nn(params_nn_a, X_train, y_train))
# Report the RMSE of validation set
RMSE_nn_val = get_RMSE_nn(params_nn_a, X_val, y_val)
print(RMSE_nn_val)

# nn fit with pretrained parameter initialization
params_nn_b = fit_nn(X_train, y_train, alpha, init=pretrain_init)
# Report the RMSE of training set
print(get_RMSE_nn(params_nn_b, X_train, y_train))
# Report the RMSE of validation set
print(get_RMSE_nn(params_nn_b, X_val, y_val))


print("####################")
print("Question 5:")

from scipy.stats import norm
from tqdm import tqdm

def train_nn_reg(X, yy, alpha, init):
    '''
    train a nn for given data with fit_nn and get the RMSE for X and yy

     Inputs:
              X N,D design matrix of input features
             yy N,  real-valued targets
            alpha   scalar regularization for weights
             init   a four-element tuple containing the initialization for parameters
     Outputs:
            RMSE   RMSE of given data for fitted nn
    '''
    # fit the neural network (fit_nn from Q4)
    params = fit_nn(X_train, y_train, alpha, init)
    # calculate the RMSE on X and yy (get_RMSE_nn from Q4)
    return get_RMSE_nn(params, X, yy)

def getPI(mu_a, cov_a, yy):
    '''
    Obtain the probability of improvement (PI) value for given unobserved alphas
     and the observed y values

     PI = Phi((mu(a)-max(yy))/sigma(a)), 
     
     Phi is the cumulative density function of standard Gaussian.
     Inputs:
           mu_a  M,   mean values for the unobserved alphas
           cov_a M,M  covariance matrix for the unobserved alphas
            yy   M,   y values for the observed alphas
     Outputs:
            PI        the PI value
    '''
    return norm.cdf((mu_a - np.max(yy)) / np.sqrt(np.diag(cov_a)))

def bayes_optim(a_rest, a_obs, RMSE_base, init=pretrain_init, num_iter=5):
    '''
    Apply a bayes optimization on a neural network to search for the best
     regularization constant, alpha

     Inputs:
          a_rest C   design matrix of input features
           a_obs T-C real-valued targets
       RMSE_base   scalar baseline RMSE
            init   a four-element tuple containing the initialization for nn parameters
                    default: pretrain_init from Q4
        num_iter   the iteration go through before returning the best alpha
                    default: 5
     Outputs:
      best_alpha   the best alpha generated by bayes optimization
             PIs   the maximum probability of improvement (PI) for each iteration
      new_alphas   corresponding alphas for each iteration 
    '''
    # Obtain the y values for the beginning observed alphas
    y_obs = []
    for i in tqdm(range(len(a_obs))):
        y_obs.append(np.log(RMSE_base) - np.log(train_nn_reg(X_val, y_val, a_obs[i], init)))
    
    # Create PIs to record the max PI for each iteration
    PIs = []
    for i in tqdm(range(num_iter)):
        # Calculate the gaussian process prosterior for the rest alphas
        # (gp_post_par imported from support_code)
        rest_mu, rest_cov = gp_post_par(a_rest, a_obs, np.array(y_obs))

        # Obtain the PI values for the unobserved alphas
        PI = getPI(rest_mu, rest_cov, y_obs)

        # pick the next alpha accordding to the PI values for each of the unobserved alphas
        # and update a_obs, a_rest, and PIs
        a_next_id = np.argmax(PI)
        a_obs = np.append(a_obs, a_rest[a_next_id])
        a_rest = np.delete(a_rest, a_next_id)
        PIs.append(PI[a_next_id])

        # Observe the y value for the picked alpha
        y_obs.append(np.log(RMSE_base) - np.log(train_nn_reg(X_val, y_val, a_obs[-1], init)))
    
    # Retrieve the best alpha according by finding which gives the maximum y_obs
    assert len(a_obs) == len(y_obs)
    best_a = a_obs[np.argmax(y_obs)]

    # a_obs[3:] corresponds to the newly added alphas
    return best_a, PIs, a_obs[3:]

# Create the possible alphas on the specific range
alphas = np.arange(0, 50, 0.02)
# randomly select three beginning alphas from alphas
alpha_obs = np.random.choice(alphas, 3, replace=False)
# Report the beginning alphas
print("Beginning alphas: ", alpha_obs)
# Delete the observed alphas from the unobserved ones to get alpha_rest
alpha_rest = np.setdiff1d(alphas, alpha_obs)
# Do bayesian optimization on the given setting (RMSE_nn_val, uni_init from Q4)
best_alpha, PIs, new_alphas = bayes_optim(alpha_rest, alpha_obs, RMSE_nn_val, init=uni_init)

# Report the maximum probability of improvement together with its alpha
# for each of the five iterations
print("PIs:         ", PIs)
print("New alphas:  ", new_alphas)
# Report the best alpha
print("Best alpha:  ", best_alpha)

# Report the RMSE on validation set and test set based on the optimized alpha
print(train_nn_reg(X_val, y_val, best_alpha, uni_init))
print(train_nn_reg(X_test, y_test, best_alpha, uni_init))


print("####################")
print("Question 6:")
# possible modifications:
# 1. Adding basis functions in phi_x (feature engineering)
# 2. optimizing K (hidden layer size)
# 3. change PI
