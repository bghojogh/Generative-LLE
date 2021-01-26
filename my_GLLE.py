import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle


class My_GLLE:

    def __init__(self, X, n_neighbors=10, n_components=None, path_save="./", max_itr_reconstruction=100):
        # X: rows are features and columns are samples
        self.n_components = n_components
        self.X = X
        self.n_samples = self.X.shape[1]
        self.n_dimensions = self.X.shape[0]
        self.n_neighbors = n_neighbors
        self.neighbor_indices = None
        self.w_linearReconstruction = None
        self.W_linearEmbedding = None
        self.n_samples_outOfSample = None
        self.neighbor_indices_for_outOfSample = None
        self.w_linearReconstruction_outOfSample = None
        self.path_save = path_save
        self.max_itr_reconstruction = max_itr_reconstruction

    def fit_transform(self, calculate_again=True):
        self.find_KNN(calculate_again=calculate_again)
        self.stochastic_linear_reconstruction(max_itr=self.max_itr_reconstruction, calculate_again=calculate_again)
        X_transformed = self.linear_embedding()
        if calculate_again:
            self.save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save=self.path_save)
        else:
            X_transformed = self.load_variable(name_of_variable="X_transformed", path=self.path_save)
        return X_transformed

    def generate_again(self, Sigma_linearReconstruction=None):
        mean_of_data = self.X.mean(axis=1).reshape((-1, 1))
        for sample_index in range(self.n_samples):
            neighbor_indices_of_this_sample = self.neighbor_indices[sample_index, :].astype(int)
            X_neighbors = self.X[:, neighbor_indices_of_this_sample]
            x = self.X[:, sample_index].reshape((-1, 1))
            #### E-step:
            if Sigma_linearReconstruction is None:
                Sigma_ = self.Sigma_linearReconstruction[:, :, sample_index]
            else:
                Sigma_ = Sigma_linearReconstruction[:, :, sample_index]
            expectation_w = Sigma_.T @ X_neighbors.T @ np.linalg.inv(X_neighbors @ Sigma_ @ X_neighbors.T) @ (x - mean_of_data)
            expectation_w_wT = Sigma_ - Sigma_.T @ X_neighbors.T @ np.linalg.inv(X_neighbors @ Sigma_ @ X_neighbors.T) @ X_neighbors @ Sigma_
            #### sampling weights:  
            self.w_linearReconstruction[sample_index, :] = np.random.multivariate_normal(mean=expectation_w.ravel(), cov=expectation_w_wT, size=1)
        X_transformed = self.linear_embedding()
        return X_transformed

    def find_KNN(self, calculate_again=True):
        if calculate_again:
            self.neighbor_indices = np.zeros((self.n_samples, self.n_neighbors))
            # --- KNN:
            connectivity_matrix = KNN(X=(self.X).T, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False, n_jobs=-1)
            connectivity_matrix = connectivity_matrix.toarray()
            # --- store indices of neighbors:
            for sample_index in range(self.n_samples):
                self.neighbor_indices[sample_index, :] = np.argwhere(connectivity_matrix[sample_index, :] == 1).ravel()
            # --- save KNN:
            self.save_variable(variable=self.neighbor_indices, name_of_variable="neighbor_indices", path_to_save=self.path_save)
        else:
            self.neighbor_indices = self.load_variable(name_of_variable="neighbor_indices", path=self.path_save)

    # def linear_reconstruction(self):
    #     self.w_linearReconstruction = np.zeros((self.n_samples, self.n_neighbors))
    #     for sample_index in range(self.n_samples):
    #         neighbor_indices_of_this_sample = self.neighbor_indices[sample_index, :].astype(int)
    #         X_neighbors = self.X[:, neighbor_indices_of_this_sample]
    #         x = self.X[:, sample_index].reshape((-1, 1))
    #         ones_vector = np.ones(self.n_neighbors).reshape((-1, 1))
    #         G = ((x.dot(ones_vector.T) - X_neighbors).T).dot(x.dot(ones_vector.T) - X_neighbors)
    #         epsilon = 0.0000001
    #         G = G + (epsilon * np.eye(self.n_neighbors))
    #         numinator = (np.linalg.inv(G)).dot(ones_vector)
    #         denominator = (ones_vector.T).dot(np.linalg.inv(G)).dot(ones_vector)
    #         self.w_linearReconstruction[sample_index, :] = ((1 / denominator) * numinator).ravel()

    def stochastic_linear_reconstruction(self, max_itr=100, calculate_again=True):
        if calculate_again:
            self.w_linearReconstruction = np.zeros((self.n_samples, self.n_neighbors))
            mean_of_data = self.X.mean(axis=1).reshape((-1, 1))
            self.Sigma_linearReconstruction = np.zeros((self.n_neighbors, self.n_neighbors, self.n_samples))
            for sample_index in range(self.n_samples):
                self.Sigma_linearReconstruction[:, :, sample_index] = np.eye(self.n_neighbors)
            for iteration_index in range(max_itr):
                print("iteration: {} in linear reconstruction".format(iteration_index))
                S1 = np.zeros((self.n_dimensions, self.n_dimensions))
                S2 = np.zeros((self.n_neighbors, self.n_neighbors))
                for sample_index in range(self.n_samples):
                    neighbor_indices_of_this_sample = self.neighbor_indices[sample_index, :].astype(int)
                    X_neighbors = self.X[:, neighbor_indices_of_this_sample]
                    x = self.X[:, sample_index].reshape((-1, 1))
                    #### E-step:
                    Sigma_ = self.Sigma_linearReconstruction[:, :, sample_index]
                    expectation_w = Sigma_.T @ X_neighbors.T @ np.linalg.inv(X_neighbors @ Sigma_ @ X_neighbors.T) @ (x - mean_of_data)
                    expectation_w_wT = Sigma_ - Sigma_.T @ X_neighbors.T @ np.linalg.inv(X_neighbors @ Sigma_ @ X_neighbors.T) @ X_neighbors @ Sigma_
                    #### sampling weights:  
                    self.w_linearReconstruction[sample_index, :] = np.random.multivariate_normal(mean=expectation_w.ravel(), cov=expectation_w_wT, size=1)
                    #### M-step:
                    S1 += ((x - mean_of_data) @ (x - mean_of_data).T) - (2 * X_neighbors @ expectation_w @ (x - mean_of_data).T) + (X_neighbors @ expectation_w_wT @ X_neighbors.T)
                    S2 += expectation_w_wT
                S1 *= (1/self.n_samples)
                S2 *= (1/self.n_samples)
                for sample_index in range(self.n_samples):
                    sigma_small = (1/(self.n_dimensions + self.n_neighbors)) * (np.trace(np.linalg.inv(X_neighbors @ X_neighbors.T) @ S1) + np.trace(S2))
                    Sigma_ = sigma_small * np.eye(self.n_neighbors)
                    self.Sigma_linearReconstruction[:, :, sample_index] = Sigma_.copy()
            self.save_variable(variable=self.w_linearReconstruction, name_of_variable="w_linearReconstruction", path_to_save=self.path_save)
            self.save_variable(variable=self.Sigma_linearReconstruction, name_of_variable="Sigma_linearReconstruction", path_to_save=self.path_save)
        else:
            self.w_linearReconstruction = self.load_variable(name_of_variable="w_linearReconstruction", path=self.path_save)
            self.Sigma_linearReconstruction = self.load_variable(name_of_variable="Sigma_linearReconstruction", path=self.path_save)

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

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))