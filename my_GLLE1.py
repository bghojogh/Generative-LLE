import numpy as np
from my_LLE import My_LLE


class My_GLLE1:

    def __init__(self, X, n_neighbors=10, n_components=None, path_save="./", max_itr=100):
        # X: rows are features and columns are samples
        self.n_components = n_components
        self.X = X
        self.n_samples = self.X.shape[1]
        self.n_dimensions = self.X.shape[0]
        self.n_neighbors = n_neighbors
        self.path_save = path_save
        self.max_itr = max_itr

    def fit_transform(self, calculate_again=True):
        ###### Notations and initializations:
        my_LLE = My_LLE(X=self.X, n_neighbors=self.n_neighbors, n_components=self.n_components)
        Y_LLE = my_LLE.fit_transform()
        w_LLE = my_LLE.w_linearReconstruction
        w = np.zeros((self.n_samples * self.n_neighbors,1))
        y = np.zeros((self.n_samples * self.n_components,1))
        x = np.zeros((self.n_samples * self.n_dimensions,1))
        X_neighbors_diagonal = np.zeros((self.n_samples * self.n_dimensions, self.n_samples * self.n_neighbors))
        Y_neighbors_diagonal = np.zeros((self.n_samples * self.n_components, self.n_samples * self.n_neighbors))
        for sample_index in range(self.n_samples):
            w[(sample_index*self.n_neighbors):((sample_index+1)*self.n_neighbors)] = w_LLE[sample_index, :].reshape((-1, 1))
            y[(sample_index*self.n_components):((sample_index+1)*self.n_components)] = Y_LLE[:, sample_index].reshape((-1, 1))
            x[(sample_index*self.n_dimensions):((sample_index+1)*self.n_dimensions)] = self.X[:, sample_index].reshape((-1, 1))
            neighbor_indices_of_this_sample = my_LLE.neighbor_indices[sample_index, :].astype(int)
            X_neighbors = self.X[:, neighbor_indices_of_this_sample]
            X_neighbors_diagonal[(sample_index*self.n_dimensions):((sample_index+1)*self.n_dimensions), 
                                 (sample_index*self.n_neighbors):((sample_index+1)*self.n_neighbors)] = X_neighbors
            Y_neighbors = Y_LLE[:, neighbor_indices_of_this_sample]
            Y_neighbors_diagonal[(sample_index*self.n_components):((sample_index+1)*self.n_components), 
                                 (sample_index*self.n_neighbors):((sample_index+1)*self.n_neighbors)] = Y_neighbors
        ###### Other notations:
        Psi_x = np.eye(self.n_samples*self.n_dimensions)
        Lambda_x = X_neighbors_diagonal
        Phi_ = np.eye(self.n_samples*self.n_neighbors)
        Psi_y = np.linalg.inv((Y_neighbors_diagonal.T @ Y_neighbors_diagonal) + Phi_)
        Lambda_y = Psi_y @ Y_neighbors_diagonal.T
        ###### Gibbs sampling:
        for iteration_index in range(self.max_itr):
            print("iteration: {} in Gibbs sampling".format(iteration_index))
            #### sampling w:
            cov_w = np.linalg.inv(((Lambda_x.T) @ np.linalg.inv(Psi_x) @ Lambda_x) + np.linalg.inv(Psi_y))
            mean_w = cov_w @ (((Lambda_x.T) @ np.linalg.inv(Psi_x) @ x) + (np.linalg.inv(Psi_y) @ Lambda_y @ y))
            w = np.random.multivariate_normal(mean=mean_w.ravel(), cov=cov_w, size=1).T
            #### sampling y:
            cov_y = np.linalg.inv(((Lambda_y.T) @ np.linalg.inv(Psi_y) @ Lambda_y) + np.eye(self.n_samples*self.n_components))
            mean_y = cov_y @ (Lambda_y.T) @ np.linalg.inv(Psi_y) @ w
            y = np.random.multivariate_normal(mean=mean_y.ravel(), cov=cov_y, size=1).T
        ###### reshaping y:
        Y_embedding = np.zeros((self.n_components, self.n_samples))
        for sample_index in range(self.n_samples):
            Y_embedding[:, sample_index] = y[(sample_index*self.n_components):((sample_index+1)*self.n_components)].ravel()
        return Y_embedding
        
    