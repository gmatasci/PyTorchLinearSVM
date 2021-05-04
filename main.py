"""
Linear SVM (soft margin) formulated and trained in PyTorch (with hyper-parameter search in scikit-learn) on a 2D toy dataset.
Inspiration from:
    - Python Engineer: https://youtu.be/UX0f9BNBcsY
    - A Developer Diary: http://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/
Giona Matasci - giona.matasci@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import relu
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator


DTYPE = torch.FloatTensor


class LinearSVM(nn.Module, BaseEstimator):
    """
    Defines a linear SVM model to be trained by gradient descent (Adam optimizer) in PyTorch.
    """
    def __init__(self, C=1.0, lr=1e-2, nr_epochs=10, plot_figs=False, plot_last_only=False, verbose=False):
        """
        Args:
            C (float): C parameter of a linear SVM.
            lr (float): Learning rate for the Adam optimizer.
            nr_epochs (int): Number of epochs to train the SVM for.
            plot_figs (bool): Flag controlling the plotting of figures.
            plot_last_only (bool): Flag to plot SVM margin and support vectors only after the model is trained.
            verbose (bool): Flag for a verbose output.
        """
        super().__init__()

        # Initialize hyper-parameters
        self.C = C
        self.lr = lr
        self.nr_epochs = nr_epochs
        self.plot_figs = plot_figs
        self.plot_last_only = plot_last_only
        self.verbose = verbose

        # Initialize model parameters and data with placeholders
        self.n = None
        self.d = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.X = None
        self.y = None

    def _hinge_loss(self, y_dec_f, y):
        """
        Computes SVM's hinge loss based on a decision function vector y_dec_f and a ground truth vector y.
        Args:
            y_dec_f (numpy.ndarray or torch.Tensor): Decision function vector.
            y (numpy.ndarray or torch.Tensor): Ground truth vector.

        Returns:
            torch.Tensor: Loss function value (a scalar).
        """
        return torch.norm(self.w**2)/2 + self.C * torch.sum(relu(1 - y*y_dec_f))

    def forward(self, X):
        """
        Computes the decision function for each sample in a given dataset X.
        Args:
            X (numpy.ndarray or torch.Tensor): Data tensor X with predictor variables.

        Returns:
            torch.Tensor: Decision function values for each sample.
        """
        if isinstance(X, (np.ndarray, np.generic)):
            X = torch.from_numpy(X).type(DTYPE)
        return X.matmul(self.w) + self.b

    def fit(self, X, y):
        """
        Fits a linear SVM model to data X with labels y.
        Args:
            X (numpy.ndarray): Data X with predictor variables.
            y (numpy.ndarray): Ground truth vector.

        Returns:
            LinearSVM: Trained PyTorchLinearSVM model object.

        """

        # Set predictor variables and ground truth label vector
        self.X = torch.from_numpy(X).type(DTYPE)
        self.y = torch.from_numpy(y).type(DTYPE)
        self.n, self.d = X.shape

        # Define the 2 trainable parameters w and b (to be set as nn.Parameter to have them be updated by the optimizer)
        self.w = nn.Parameter(Variable(torch.randn(self.d).type(DTYPE), requires_grad=True))  # same dimension as the dataset
        self.b = nn.Parameter(Variable(torch.randn(1).type(DTYPE), requires_grad=True))

        if self.plot_figs:
            self.plot_margin(title='Randomly initialized hyperplane')

        optimizer = Adam(self.parameters(), lr=self.lr)

        # Train linear SVM by gradient descent (Adam) for self.nr_epochs epochs
        for epoch in range(1, self.nr_epochs+1):

            # Compute decision function values for the training set
            y_dec_f = self.forward(self.X)   # alternatively, calling self() would also work

            # Compute loss per sample
            loss_total = self._hinge_loss(y_dec_f, self.y)
            loss = loss_total/self.n

            # Compute gradients and update parameters by backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                print(f'Epoch {epoch}, loss: {loss}')

            if self.plot_figs and epoch % 10 == 0:
                self.plot_margin(title=f'Hyperplane at epoch {epoch}')

        # Get indices of support vectors (in case of soft-margin SVM it is the samples inside or beyond the margin)
        self.support_vectors = np.where(self.y.detach().numpy() * self.forward(self.X).detach().numpy() <= 1)[0]
        if self.plot_last_only:
            self.plot_margin(title='Final hyperplane')

        return self

    def predict(self, X):
        """
        Runs prediction step on input data X.
        Args:
            X (numpy.ndarray): Data X with predictor variables.

        Returns:
            numpy.ndarray: Binary prediction for each input sample (1 or -1).
        """
        return torch.sign(self.forward(X)).detach().numpy()

    def plot_margin(self, title=''):
        """
        Plots training data points, the SVM decision function and support vectors (training samples inside or beyond the margin).
        Args:
            title (str): Custom title of the plot.
        """
        fig, ax = plt.subplots(dpi=200)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap=plt.cm.Paired, alpha=.7)

        plt.title(f'{title}, C: {self.C}')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.forward(xy).reshape(XX.shape).detach().numpy()

        # Plot decision function and margins
        ax.contour(XX, YY, Z,
                   colors=['b', 'k', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        # Highlight the support vectors
        if self.support_vectors is not None:
            ax.scatter(self.X[self.support_vectors, 0], self.X[self.support_vectors, 1], s=100,
                       linewidth=1, facecolors='none', edgecolors='k')

        plt.show()


if __name__ == '__main__':

    # Define hyper-parameters
    cfg = {}
    cfg['nr_epochs'] = 200
    cfg['C_vect'] = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

    # Create toy dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=1.5, random_state=0)
    y = 2*y - 1  # put labels as either +1 or -1 (requirement of SVM formulation)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Normalize both sets with mean and std dev of training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create model object
    model = LinearSVM(nr_epochs=cfg['nr_epochs'], plot_last_only=True)

    # Find best C parameter by cross-validation (5-fold)
    parameters = {'C': cfg['C_vect']}
    gs = GridSearchCV(model, parameters, scoring='accuracy', cv=5)
    gs.fit(X_train, y_train)
    print(f'CV results:\nbest params: {gs.best_params_}, best accuracy: {gs.best_score_:.3f}')

    # Predict on test set using the best model and assess performance
    y_test_pred = gs.predict(X=X_test)
    cr = classification_report(y_test, y_test_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    print(f'Results on test set:\n{cr_df}')
