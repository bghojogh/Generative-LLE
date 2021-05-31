from functions.my_LLE import My_LLE
from functions.my_GLLE import My_GLLE
from functions.my_GLLE_DirectSampling import My_GLLE_DirectSampling
import functions.load_datasets
import functions.utils as utils
from sklearn import manifold, datasets
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # settings:
    method = "GLLE"  #--> LLE_ready, LLE, GLLE, GLLE_DirectSampling
    dataset = "Sphere"  #--> Swiss_roll, Swiss_roll_hole, S_curve, Sphere, Sphere_small, digits, MNIST, ORL_glasses
    make_dataset_again = False
    embed_again = True
    generate_embedding_again = False
    plot_manifold_interpolation = False
    n_generation_of_embedding = 30
    max_iterations = 10
    n_components = 5

    if make_dataset_again:
        labels, color = None, None
        if dataset == "Swiss_roll":
            # X, color = datasets.make_swiss_roll(n_samples=1500)
            X, color = datasets.make_swiss_roll(n_samples=5000)
        if dataset == "Swiss_roll_hole":
            X, color = load_datasets.make_swiss_roll_with_hole(n_samples=4950)
            # utils.plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")
        elif dataset == "S_curve":
            # X, color = datasets.make_s_curve(n_samples=1500, random_state=0)
            X, color = datasets.make_s_curve(n_samples=5000, random_state=0)
        elif dataset == "Sphere":
            X, color = load_datasets.make_sphere_dataset(n_samples=5000, severed_poles=True)
        elif dataset == "Sphere_small":
            X, color = load_datasets.make_sphere_dataset(n_samples=1000, severed_poles=True)
        elif dataset == "digits":
            # https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
            # https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
            # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
            # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
            digits = datasets.load_digits(n_class=6)
            X = digits.data
            labels = digits.target
            # scaler = StandardScaler()
            # X = scaler.fit_transform(X)
        elif dataset == "MNIST":
            _, labels, X = load_datasets.read_MNIST_dataset(MNIST_subset_cardinality_training=100, read_dataset_again=True)
            X = X.T  #--> make it row-wise
        elif dataset == "ORL_glasses":
            X, labels = load_datasets.read_ORL_glasses_dataset(scale=0.5)
            X = X.T  #--> make it row-wise
        if dataset in ["Swiss_roll", "Swiss_roll_hole", "S_curve", "Sphere", "Sphere_small"]:
            utils.plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")
        utils.save_variable(variable=X, name_of_variable="X", path_to_save='./datasets/'+dataset+"/")
        if color is not None:
            utils.save_variable(variable=color, name_of_variable="color", path_to_save='./datasets/'+dataset+"/")
        if labels is not None:
            utils.save_variable(variable=labels, name_of_variable="labels", path_to_save='./datasets/'+dataset+"/")
        if dataset == "digits":
            utils.save_variable(variable=digits, name_of_variable="digits", path_to_save='./datasets/'+dataset+"/")
    else:
        X = utils.load_variable(name_of_variable="X", path='./datasets/'+dataset+"/")
        try:
            color = utils.load_variable(name_of_variable="color", path='./datasets/'+dataset+"/")
            utils.plot_3D(X, color, path_to_save='./datasets/'+dataset+"/", name="dataset")
        except:
            color = None
        try:
            labels = utils.load_variable(name_of_variable="labels", path='./datasets/'+dataset+"/")
        except:
            labels = None
        if dataset == "digits":
            digits = utils.load_variable(name_of_variable="digits", path='./datasets/'+dataset+"/")


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
    if dataset in ["Swiss_roll", "Swiss_roll_hole", "S_curve", "Sphere", "Sphere_small"]:
        # utils.plot_3D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_3D")
        utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")
    elif dataset in ["digits", "MNIST", "ORL_glasses"]:
        utils.plot_embedding_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_numbers")  
        utils.plot_2D_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")
        # utils.plot_embedding_with_labels_and_images(Y, labels, images=digits.images)
        # utils.plot_components(Y.T, labels, which_dimensions_to_plot=[0,1], images=digits.images, image_scale=2, markersize=10, thumb_frac=0.05, cmap='gray')

    if (method == "GLLE" or method == "GLLE_DirectSampling") and generate_embedding_again:
        for itr in range(n_generation_of_embedding):
            if method == "GLLE":
                X_transformed = my_GLLE.generate_again()
            elif method == "GLLE_DirectSampling":
                X_transformed = my_GLLE_DirectSampling.generate_again()
            Y = X_transformed.T
            if color is not None:
                utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/generation/", name="embedding_gen"+str(itr))
            if labels is not None:
                utils.plot_embedding_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/generation/", name="embedding_numbers_gen"+str(itr)) 
                utils.plot_2D_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/generation/", name="embedding_gen"+str(itr))
            utils.save_variable(variable=X_transformed, name_of_variable="X_transformed", path_to_save="./saved_files/"+method+"/"+dataset+"/generation/gen"+str(itr)+"/")

    if (method == "GLLE" or method == "GLLE_DirectSampling") and plot_manifold_interpolation:
        # n_interpolation = 5
        # grid_ = np.linspace(0.01, 10, n_interpolation)
        grid_ = [0.01, 0.1, 1, 5, 10]
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
        


if __name__ == "__main__":
    main()