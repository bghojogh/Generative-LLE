import numpy as np
from sklearn.utils import check_random_state
from matplotlib import gridspec
import os
import pickle
import functions.utils as utils
from sklearn.preprocessing import StandardScaler
from PIL import Image
from skimage.transform import resize


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

def make_swiss_roll_with_hole(n_samples=100, noise=0.0, random_state=None):
    # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/datasets/_samples_generator.py#L1401
    generator = check_random_state(random_state)
    t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
    x = t * np.cos(t)
    y = 21 * generator.rand(1, n_samples)
    z = t * np.sin(t)

    X = np.concatenate((x, y, z))

    # ranges of three coordinates:
    # np.max(X[2,:]) = 14.136094274051034
    # np.min(X[2,:]) = -11.040707948365629
    # np.max(X[1,:]) = 20.996180119068676
    # np.min(X[1,:]) = 0.004955591716158225
    # np.max(X[0,:]) = 12.60593128332993
    # np.min(X[0,:]) = -9.47727415039662

    # mask = ((X[0, :]>-8) & (X[0, :]<0) & (X[2, :]>-4) & (X[2, :]<7) & (X[1, :]>5) & (X[1, :]<15))
    # mask = ((X[0, :]>-10) & (X[0, :]<0) & (X[2, :]>-4) & (X[2, :]<7) & (X[1, :]>5) & (X[1, :]<15))
    mask = ((X[0, :]>-10) & (X[0, :]<-5) & (X[2, :]>-4) & (X[2, :]<7) & (X[1, :]>5) & (X[1, :]<15))
    X[:, mask] = None
    
    X += noise * generator.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)

    # remove the nan values (related to hole):
    hole_point_indices_3D = np.isnan(X)
    hole_point_indices = np.asarray([(hole_point_indices_3D[i, 0] | hole_point_indices_3D[i, 1] | hole_point_indices_3D[i, 2]) for i in range(hole_point_indices_3D.shape[0])])
    t = t[~hole_point_indices]
    X = X[~hole_point_indices]

    return X, t

def read_MNIST_dataset(MNIST_subset_cardinality_training, read_dataset_again=True):
    # MNIST_subset_cardinality_training --> is per class
    subset_of_MNIST = True
    pick_subset_of_MNIST_again = True
    MNIST_subset_cardinality_testing = 10
    path_dataset_save = './datasets/MNIST/'
    file = open(path_dataset_save+'X_train.pckl','rb')
    X_train = pickle.load(file); file.close()
    file = open(path_dataset_save+'y_train.pckl','rb')
    y_train = pickle.load(file); file.close()
    file = open(path_dataset_save+'X_test.pckl','rb')
    X_test = pickle.load(file); file.close()
    file = open(path_dataset_save+'y_test.pckl','rb')
    y_test = pickle.load(file); file.close()
    dataset = "MNIST_" + str(MNIST_subset_cardinality_training)
    if subset_of_MNIST:
        if pick_subset_of_MNIST_again and read_dataset_again:
            dimension_of_data = 28 * 28
            X_train_picked = np.empty((0, dimension_of_data))
            y_train_picked = np.empty((0, 1))
            for label_index in range(10):
                X_class = X_train[y_train == label_index, :]
                X_class_picked = X_class[0:MNIST_subset_cardinality_training, :]
                X_train_picked = np.vstack((X_train_picked, X_class_picked))
                y_class = y_train[y_train == label_index]
                y_class_picked = y_class[0:MNIST_subset_cardinality_training].reshape((-1, 1))
                y_train_picked = np.vstack((y_train_picked, y_class_picked))
            y_train_picked = y_train_picked.ravel()
            X_test_picked = np.empty((0, dimension_of_data))
            y_test_picked = np.empty((0, 1))
            for label_index in range(10):
                X_class = X_test[y_test == label_index, :]
                X_class_picked = X_class[0:MNIST_subset_cardinality_testing, :]
                X_test_picked = np.vstack((X_test_picked, X_class_picked))
                y_class = y_test[y_test == label_index]
                y_class_picked = y_class[0:MNIST_subset_cardinality_testing].reshape((-1, 1))
                y_test_picked = np.vstack((y_test_picked, y_class_picked))
            y_test_picked = y_test_picked.ravel()
            # X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
            # X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
            # y_train_picked = y_train[0:MNIST_subset_cardinality_training]
            # y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
            utils.save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save+dataset+"/")
            utils.save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save+dataset+"/")
            utils.save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save+dataset+"/")
            utils.save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save+dataset+"/")
        else:
            file = open(path_dataset_save+dataset+"/"+'X_train_picked.pckl','rb')
            X_train_picked = pickle.load(file); file.close()
            file = open(path_dataset_save+dataset+"/"+'X_test_picked.pckl','rb')
            X_test_picked = pickle.load(file); file.close()
            file = open(path_dataset_save+dataset+"/"+'y_train_picked.pckl','rb')
            y_train_picked = pickle.load(file); file.close()
            file = open(path_dataset_save+dataset+"/"+'y_test_picked.pckl','rb')
            y_test_picked = pickle.load(file); file.close()
        X_train = X_train_picked
        X_test = X_test_picked
        y_train = y_train_picked
        y_test = y_test_picked
    data = X_train.T / 255
    data_test = X_test.T / 255
    labels = y_train.reshape((1, -1))
    n_samples = data.shape[1]
    image_height = 28
    image_width = 28
    # ---- normalize (standardation):
    X_notNormalized = data
    # data = data / 255
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)  #--> comment it for LLE on MNIST
    data = (scaler.transform(data.T)).T   #--> comment it for LLE on MNIST
    X = data
    y = labels
    y = y.ravel()
    # print(np.max(X))
    # input("hi")
    return X, y, X_notNormalized

def read_ORL_glasses_dataset(scale=1):
    path_dataset = "./datasets/ORL_glasses/"
    n_samples = 400
    image_height = int(112 * scale)
    image_width = int(92 * scale)
    data = np.zeros((image_height * image_width, n_samples))
    labels = np.zeros((1, n_samples))
    image_index = -1
    for class_index in range(2):
        for filename in os.listdir(path_dataset + "class" + str(class_index + 1) + "/"):
            image_index = image_index + 1
            if image_index >= n_samples:
                break
            img = load_image(address_image=path_dataset + "class" + str(class_index + 1) + "/" + filename,
                                image_height=image_height, image_width=image_width, do_resize=False, scale=scale)
            data[:, image_index] = img.ravel()
            labels[:, image_index] = class_index
    # ---- cast dataset from string to float:
    data = data.astype(np.float)
    # ---- normalize (standardation):
    X_notNormalized = data
    # data = data / 255
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
    data = (scaler.transform(data.T)).T
    X = data
    y = labels.ravel()
    return X, y

def load_image(address_image, image_height, image_width, do_resize=False, scale=1):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    if do_resize:
        size = int(image_height * scale), int(image_width * scale)
        # img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img)
    img_arr = resize(img_arr, (int(img_arr.shape[0]*scale), int(img_arr.shape[1]*scale)), order=5, preserve_range=True, mode="constant")
    return img_arr