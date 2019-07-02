import scipy.io
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate, KFold
from sklearn import svm
from skfeature.function.information_theoretical_based import MRMR


# load data
mat = scipy.io.loadmat('data/colon.mat')
X = mat['X']    # data  # (62, 2000)
X = X.astype(float)
y = mat['Y']    # label
y = y[:, 0]     # (62,)
#n_samples, n_features = X.shape    # number of samples and number of features

# perform evaluation on classification task
num_fea = 10    # number of selected features

# obtain the index of each feature on the training set
'''
Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
'''
idx,_,_ = MRMR.mrmr(X, y, n_selected_features=num_fea)

# obtain selected features
selected_features = X[:, idx[0:num_fea]]
