from my_LLE import My_LLE
from my_GLLE import My_GLLE
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
from matplotlib import gridspec
import numpy as np
from sklearn.utils import check_random_state


def main():
    # settings:
    method = "LLE"  #--> LLE_ready, LLE, GLLE
    dataset = "Sphere"  #--> Swiss_roll, S_curve, Sphere
    make_dataset_again = False
    embed_again = True
    generate_embedding_again = False
    plot_manifold_interpolation = False
    n_generation_of_embedding = 10
    max_itr_reconstruction = 10
    n_components = 5

    if make_dataset_again:
        if dataset == "Swiss_roll":
            # X, color = datasets.make_swiss_roll(n_samples=1500)
            X, color = datasets.make_swiss_roll(n_samples=5000)
        elif dataset == "S_curve":
            # X, color = datasets.make_s_curve(n_samples=1500, random_state=0)
            X, color = datasets.make_s_curve(n_samples=5000, random_state=0)
        elif dataset == "Sphere":
            X, color = make_sphere_dataset(n_samples=5000, severed_poles=True)
        plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")
        save_variable(variable=X, name_of_variable="X", path_to_save='./datasets/'+dataset+"/")
        save_variable(variable=color, name_of_variable="color", path_to_save='./datasets/'+dataset+"/")
    else:
        X = load_variable(name_of_variable="X", path='./datasets/'+dataset+"/")
        color = load_variable(name_of_variable="color", path='./datasets/'+dataset+"/")
        plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")

    if method == "LLE_ready":
        # https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding
        # Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components)  
        Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components, eigen_solver="dense")  
    elif method == "LLE":
        my_LLE = My_LLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/LLE/"+dataset+"/")
        Y = my_LLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    elif method == "GLLE":
        my_GLLE = My_GLLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/GLLE/"+dataset+"/", max_itr_reconstruction=max_itr_reconstruction)
        Y = my_GLLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    # plot_3D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_3D")
    plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")

    if method == "GLLE" and generate_embedding_again:
        for itr in range(n_generation_of_embedding):
            X_transformed = my_GLLE.generate_again()
            Y = X_transformed.T
            plot_2D(Y, color, path_to_save="./saved_files/GLLE/"+dataset+"/generation/", name="embedding_gen"+str(itr))
            save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save="./saved_files/GLLE/"+dataset+"/generation/gen"+str(itr)+"/")

    if method == "GLLE" and plot_manifold_interpolation:
        n_interpolation = 10
        # grid_ = np.linspace(-3, 3, n_interpolation)
        grid_ = np.linspace(-20, 20, n_interpolation)
        for itr, sigma_i_multiplication in enumerate(grid_):
            Sigma_linearReconstruction = my_GLLE.Sigma_linearReconstruction[:, :, :] * sigma_i_multiplication
            X_transformed = my_GLLE.generate_again(Sigma_linearReconstruction)
            Y = X_transformed.T
            plot_2D(Y, color, path_to_save="./saved_files/GLLE/"+dataset+"/interpolation/", name="embedding_gen"+str(itr), title="sigma_multipler = "+str(sigma_i_multiplication))
            save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save="./saved_files/GLLE/"+dataset+"/interpolation/itr"+str(itr)+"/")
        

def plot_3D(X, color, path_to_save="./", name="temp"):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.xticks([]), plt.yticks([]), ax.set_zticks([])
    plt.savefig(path_to_save+name+".png")
    # plt.show()

def plot_2D(X, color, path_to_save="./", name="temp", title=None):
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    ax = plt.axes()
    ax.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()

def make_sphere_dataset(n_samples, severed_poles=False):
    #### https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html
    # Create our sphere.
    random_state = check_random_state(0)
    # p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    p = random_state.rand(n_samples) * (2 * np.pi - 1)
    t = random_state.rand(n_samples) * np.pi
    # Sever the poles from the sphere.
    if severed_poles:
        # indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
        indices = ((t < (np.pi - (np.pi / 5))) & (t > ((np.pi / 3))))
        colors = p[indices]
        x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
            np.sin(t[indices]) * np.sin(p[indices]), \
            np.cos(t[indices])
    else:
        colors = p
        x, y, z = np.sin(t) * np.cos(p), \
            np.sin(t) * np.sin(p), \
            np.cos(t)
    X = np.array([x, y, z]).T
    return X, colors

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

if __name__ == "__main__":
    main()