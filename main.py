from my_LLE import My_LLE
from my_GLLE import My_GLLE
from my_GLLE_DirectSampling import My_GLLE_DirectSampling
from sklearn import manifold, datasets
import os
import pickle
import utils
from matplotlib import gridspec
import numpy as np
from sklearn.utils import check_random_state


def main():
    # settings:
    method = "GLLE_DirectSampling"  #--> LLE_ready, LLE, GLLE, GLLE_DirectSampling
    dataset = "Swiss_roll"  #--> Swiss_roll, S_curve, Sphere, Sphere_small
    make_dataset_again = False
    embed_again = False
    generate_embedding_again = False
    plot_manifold_interpolation = True
    n_generation_of_embedding = 30
    max_iterations = 10
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
        elif dataset == "Sphere_small":
            X, color = make_sphere_dataset(n_samples=1000, severed_poles=True)
        utils.plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")
        utils.save_variable(variable=X, name_of_variable="X", path_to_save='./datasets/'+dataset+"/")
        utils.save_variable(variable=color, name_of_variable="color", path_to_save='./datasets/'+dataset+"/")
    else:
        X = utils.load_variable(name_of_variable="X", path='./datasets/'+dataset+"/")
        color = utils.load_variable(name_of_variable="color", path='./datasets/'+dataset+"/")
        utils.plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")

    if method == "LLE_ready":
        # https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding
        # Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components)  
        Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components, eigen_solver="dense")  
    elif method == "LLE":
        my_LLE = My_LLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/")
        Y = my_LLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    elif method == "GLLE":
        my_GLLE = My_GLLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/", max_itr_reconstruction=max_iterations)
        Y = my_GLLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    elif method == "GLLE_DirectSampling":
        my_GLLE_DirectSampling = My_GLLE_DirectSampling(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/", max_itr=max_iterations)
        Y = my_GLLE_DirectSampling.fit_transform(calculate_again=embed_again)
        Y = Y.T
    # utils.plot_3D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_3D")
    utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")

    if (method == "GLLE" or method == "GLLE_DirectSampling") and generate_embedding_again:
        for itr in range(n_generation_of_embedding):
            if method == "GLLE":
                X_transformed = my_GLLE.generate_again()
            elif method == "GLLE_DirectSampling":
                X_transformed = my_GLLE_DirectSampling.generate_again()
            Y = X_transformed.T
            utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/generation/", name="embedding_gen"+str(itr))
            utils.save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save="./saved_files/"+method+"/"+dataset+"/generation/gen"+str(itr)+"/")

    if (method == "GLLE" or method == "GLLE_DirectSampling") and plot_manifold_interpolation:
        n_interpolation = 10
        # grid_ = np.linspace(-3, 3, n_interpolation)
        grid_ = np.linspace(-20, 20, n_interpolation)
        for itr, sigma_i_multiplication in enumerate(grid_):
            if method == "GLLE":
                Sigma_linearReconstruction = my_GLLE.Sigma_linearReconstruction[:, :, :] * sigma_i_multiplication
                X_transformed = my_GLLE.generate_again(Sigma_linearReconstruction)
            elif method == "GLLE_DirectSampling":
                Sigma_linearReconstruction = my_GLLE_DirectSampling.Cov_weights_linearReconstruction[:, :, :] * sigma_i_multiplication
                X_transformed = my_GLLE_DirectSampling.generate_again(Sigma_linearReconstruction)
            Y = X_transformed.T
            utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/interpolation/", name="embedding_gen"+str(itr), title="sigma_multipler = "+str(sigma_i_multiplication))
            utils.save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save="./saved_files/"+method+"/"+dataset+"/interpolation/itr"+str(itr)+"/")
        

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


if __name__ == "__main__":
    main()