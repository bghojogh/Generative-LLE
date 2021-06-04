from functions.my_LLE import My_LLE
from functions.my_GLLE import My_GLLE
from functions.my_GLLE_DirectSampling import My_GLLE_DirectSampling
import functions.load_datasets as load_datasets
import functions.utils as utils
from sklearn import manifold, datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt

import json
with open('settings.json') as json_file:
    settings = json.load(json_file)

# ##################################### options for settings in the json file #####################################
# ================ method: 
#   LLE ---> it is original deterministic LLE
#   GLLE ---> it is GLLE with EM algorithm
#   GLLE_DirectSampling ---> it is GLLE with direct sampling
# ================ dataset:
#   Swiss_roll, Swiss_roll_hole, S_curve, Sphere, Sphere_small ---> these are ready toy datasets
#   User_data ---> User can put their dataset as a csv files named "data" (row-wise data) and an optional "colors" file
# ================ make_dataset_again:
#   True ---> generate the ready toy datasets again (with new possible data settings)
#   False ---> load the previously generated toy dataset (it will throw error if you have not generated a dataset before)
# ================ embed_again:
#   True ---> train the embedding (unfolding) again
#   False ---> do not train again and load the previous training phase. This can be useful for when user wants to generate several unfoldings and does not want to train again
# ================ generate_embedding_again:
#   True ---> generate [multiple] unfoldings (embeddings)
#   False ---> do not generate unfoldings (embeddings)
# ================ analyze_covariance_scales:
#   True ---> generate unfoldings for various scales of covariance matrix for the sake of analysis
#   False ---> do not generate unfoldings for various scales of covariance matrix
# ================ n_generation_of_embedding:
#   A positive integer ---> it is the number of unfoldings (embeddings) to generate
# ================ max_iterations:
#   A positive integer ---> maximum number of iterations for EM algorithm in stochastic linear reconstruction of GLLE
# ================ n_components:
#   A positive integer (between 1 and dimensionality of data) ---> the dimensionality of unfolding (embedding)
# ================ verbosity:
#   0 ---> do not print logging information
#   1 ---> print logging information of level one
#   2 ---> print logging information of levels one and two


def main():

    ##################################### loading settings #####################################

    method = settings["method"]
    dataset = settings["dataset"]
    make_dataset_again = True if settings["make_dataset_again"] == "True" else False
    embed_again = True if settings["embed_again"] == "True" else False
    generate_embedding_again = True if settings["generate_embedding_again"] == "True" else False
    analyze_covariance_scales = True if settings["analyze_covariance_scales"] == "True" else False
    n_generation_of_embedding = settings["n_generation_of_embedding"]
    max_iterations = settings["max_iterations"]
    n_components = settings["n_components"]
    verbosity = settings["verbosity"]

    ##################################### loading or generating dataset #####################################

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
        if dataset == "User_data":
            try: 
                X = genfromtxt("datasets/User_data/data.csv", delimiter=',')
            except Exception as ex:
                raise ValueError("There is no any user data in the dataset folder!")
            try: 
                color = genfromtxt("datasets/User_data/color.csv", delimiter=',')
            except Exception as ex:
                color = None
            try: 
                labels = genfromtxt("datasets/User_data/labels.csv", delimiter=',')
            except Exception as ex:
                labels = None
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

    ##################################### training the GLLE method #####################################

    if method == "LLE_ready":
        # https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding
        # Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components)  
        Y, err = manifold.locally_linear_embedding(X, n_neighbors=10, n_components=n_components, eigen_solver="dense")  
    elif method == "LLE":
        my_LLE = My_LLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/", verbosity=verbosity)
        Y = my_LLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    elif method == "GLLE":
        my_GLLE = My_GLLE(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/", max_itr_reconstruction=max_iterations, verbosity=verbosity)
        Y = my_GLLE.fit_transform(calculate_again=embed_again)
        Y = Y.T
    elif method == "GLLE_DirectSampling":
        my_GLLE_DirectSampling = My_GLLE_DirectSampling(X.T, n_neighbors=10, n_components=n_components, path_save="./saved_files/"+method+"/"+dataset+"/", verbosity=verbosity)
        Y = my_GLLE_DirectSampling.fit_transform(calculate_again=embed_again)
        Y = Y.T
    
    ##################################### plot the trained unfolding #####################################
    
    if dataset in ["Swiss_roll", "Swiss_roll_hole", "S_curve", "Sphere", "Sphere_small"]:
        # utils.plot_3D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_3D")
        utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")
    elif dataset in ["digits", "MNIST", "ORL_glasses"]:
        utils.plot_embedding_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_numbers")  
        utils.plot_2D_with_labels(Y, labels, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")
        # utils.plot_embedding_with_labels_and_images(Y, labels, images=digits.images)
        # utils.plot_components(Y.T, labels, which_dimensions_to_plot=[0,1], images=digits.images, image_scale=2, markersize=10, thumb_frac=0.05, cmap='gray')
    elif dataset == "User_data":
        # utils.plot_3D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding_3D")
        utils.plot_2D(Y, color, path_to_save="./saved_files/"+method+"/"+dataset+"/", name="embedding")

    ##################################### generating unfoldings #####################################

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

    ##################################### analyzing the impact of covariance scales #####################################

    if (method == "GLLE" or method == "GLLE_DirectSampling") and analyze_covariance_scales:
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