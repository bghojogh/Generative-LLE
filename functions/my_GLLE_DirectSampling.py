import numpy as np
from functions.my_LLE import My_LLE
import matplotlib.pyplot as plt
import functions.utils as utils


class My_GLLE_DirectSampling:

    def __init__(self, X, n_neighbors=10, n_components=None, path_save="./", verbosity=0):
        # X: rows are features and columns are samples
        self.n_components = n_components
        self.X = X
        self.n_samples = self.X.shape[1]
        self.n_dimensions = self.X.shape[0]
        self.n_neighbors = n_neighbors
        self.path_save = path_save
        self.w_linearReconstruction = None
        self.Cov_weights_linearReconstruction = None
        self.mean_weights_linearReconstruction = None
        self.neighbor_indices = None
        self.verbosity = verbosity

    def fit_transform(self, calculate_again=True):
        if calculate_again:
            self.stochastic_linear_reconstruction(calculate_again=calculate_again)
            if self.verbosity >= 1: print("Linear reconstruction is done...")
            X_transformed = self.linear_embedding()
            if self.verbosity >= 1: print("Linear embedding is done...")
            utils.save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save=self.path_save)
        else:
            if self.verbosity >= 1: print("Loading previous embedding...")
            X_transformed = utils.load_variable(name_of_variable="X_transformed", path=self.path_save)
        return X_transformed

    def generate_again(self, Cov_weights_linearReconstruction=None, mean_weights_linearReconstruction=None):
        if self.verbosity >= 1: print("Generating a new embedding (unfolding)...")
        for sample_index in range(self.n_samples):
            if self.verbosity >= 1 and sample_index % 1000 == 0:
                if self.verbosity >= 2: print("processing sample {}/{}".format(sample_index,self.n_samples))
            if Cov_weights_linearReconstruction is None:
                cov_w = self.Cov_weights_linearReconstruction[:, :, sample_index]
            else:
                cov_w = Cov_weights_linearReconstruction[:, :, sample_index]
            if mean_weights_linearReconstruction is None:
                mean_w = self.mean_weights_linearReconstruction[:, sample_index]
            else:
                mean_w = mean_weights_linearReconstruction[:, sample_index]
            #### sampling weights:
            self.w_linearReconstruction[sample_index, :] = np.random.multivariate_normal(mean=mean_w.ravel(), cov=cov_w, size=1)
        X_transformed = self.linear_embedding()
        return X_transformed

    def stochastic_linear_reconstruction(self, calculate_again=True):
        if calculate_again:
            my_LLE = My_LLE(X=self.X, n_neighbors=self.n_neighbors, n_components=self.n_components)
            Y_LLE = my_LLE.fit_transform()
            w_LLE = (my_LLE.w_linearReconstruction).T
            self.neighbor_indices = my_LLE.neighbor_indices
            # Phi_ = np.eye(self.n_neighbors) * 1e-10
            Phi_ = 0
            self.w_linearReconstruction = np.zeros((self.n_samples, self.n_neighbors))
            self.mean_weights_linearReconstruction = np.zeros((self.n_neighbors, self.n_samples))
            self.Cov_weights_linearReconstruction = np.zeros((self.n_neighbors, self.n_neighbors, self.n_samples))
            for sample_index in range(self.n_samples):
                self.Cov_weights_linearReconstruction[:, :, sample_index] = np.eye(self.n_neighbors)
            for sample_index in range(self.n_samples):
                if self.verbosity >= 2 and sample_index % 1000 == 0:
                    print("processing sample {}/{}".format(sample_index,self.n_samples))
                neighbor_indices_of_this_sample = self.neighbor_indices[sample_index, :].astype(int)
                X_neighbors = self.X[:, neighbor_indices_of_this_sample]
                Y_neighbors = Y_LLE[:, neighbor_indices_of_this_sample]
                #### sampling w:
                cov_w = np.linalg.inv( (X_neighbors.T @ X_neighbors) + (Y_neighbors.T @ Y_neighbors) + Phi_ )
                # mean_w = cov_w @ ( (X_neighbors.T @ x_i) + (Y_neighbors.T @ y) )
                mean_w = w_LLE[:, sample_index]
                self.w_linearReconstruction[sample_index, :] = np.random.multivariate_normal(mean=mean_w.ravel(), cov=cov_w, size=1)
                self.Cov_weights_linearReconstruction[:, :, sample_index] = cov_w
                self.mean_weights_linearReconstruction[:, sample_index] = mean_w.ravel()
            utils.save_variable(variable=self.w_linearReconstruction, name_of_variable="w_linearReconstruction", path_to_save=self.path_save)
            utils.save_variable(variable=self.Cov_weights_linearReconstruction, name_of_variable="Cov_weights_linearReconstruction", path_to_save=self.path_save)
            utils.save_variable(variable=self.mean_weights_linearReconstruction, name_of_variable="mean_weights_linearReconstruction", path_to_save=self.path_save)
            utils.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save=self.path_save)
        else:
            self.w_linearReconstruction = utils.load_variable(name_of_variable="w_linearReconstruction", path=self.path_save)
            self.Cov_weights_linearReconstruction = utils.load_variable(name_of_variable="Cov_weights_linearReconstruction", path=self.path_save)
            self.mean_weights_linearReconstruction = utils.load_variable(name_of_variable="mean_weights_linearReconstruction", path=self.path_save)
            self.neighbor_indices = utils.load_variable(name_of_variable="neighbor_indices", path=self.path_save)
        
    def linear_embedding(self):
        self.W_linearEmbedding = np.zeros((self.n_samples, self.n_samples))
        for sample_index in range(self.n_samples):
            neighbor_indices_of_this_sample = self.neighbor_indices[sample_index, :].astype(int)
            self.W_linearEmbedding[sample_index, neighbor_indices_of_this_sample] = self.w_linearReconstruction[sample_index, :].ravel()
        temp = np.eye(self.n_samples) - self.W_linearEmbedding
        M = (temp.T).dot(temp)
        eig_val, eig_vec = np.linalg.eigh(M)
        idx = eig_val.argsort()  # sort eigenvalues in ascending order (smallest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            X_transformed = eig_vec[:, 1:self.n_components+1] #--> note that first eigenvalue is zero
        else:
            X_transformed = eig_vec[:, 1:] #--> note that first eigenvalue is zero
        X_transformed = X_transformed.T  #--> the obtained Y in Laplacian eigenmap is row-wise vectors, so we transpose it
        return X_transformed