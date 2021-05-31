import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
# from skimage.transform import resize


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

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    if type(variable) is list:
        variable = np.asarray(variable)
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def plot_3D(X, color, path_to_save="./", name="temp"):
    # X: rows are samples and columns are features
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.xticks([]), plt.yticks([]), ax.set_zticks([])
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()

def plot_2D(X, color, path_to_save="./", name="temp", title=None):
    # X: rows are samples and columns are features
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

def plot_2D_with_labels(X, labels, class_names=None, path_to_save="./", name="temp", title=None):
    # X: rows are samples and columns are features
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    if class_names is None:
        labels_unique = np.unique(np.asarray(labels))
        class_names = labels_unique.astype(str)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    cmap = 'Spectral'  #--> plt.cm.brg, 'Spectral', plt.cm.Set1, plt.cm.tab10
    plt.scatter(X[:, 0], X[:, 1], s=10, c=labels, cmap=cmap, alpha=1.0)
    if title is not None:
        plt.title(title)
    n_classes = len(class_names)
    cbar = plt.colorbar(boundaries=np.arange(n_classes + 1) - 0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(class_names)
    plt.xticks([]), plt.yticks([])
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()

def plot_embedding_with_labels(X, labels, path_to_save="./", name="temp", title=None):
    # X: rows are samples and columns are features
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()

def plot_embedding_with_labels_and_images(X, labels, images, path_to_save="./", name="temp", title=None):
    # X: rows are samples and columns are features
    # images: size (n_samples, n_pixels_rows, n_pixels_columns)
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    if X.shape[1] > 2:
        X = X[:, :2]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i,:] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i,:,:], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()

def plot_components(X_projected, labels, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray', path_to_save="./", name="temp", title=None):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    if not os.path.exists(path_to_save): 
        os.makedirs(path_to_save)
    colormap = plt.cm.brg
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    # ax = ax or plt.gca()
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    ax.scatter(X_projected[:, 0], X_projected[:, 1], c=labels, cmap=colormap, edgecolors='k')
    #### images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # plot the first (original) image once more to be on top of other images:
        # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path_to_save+name+".png")
    # plt.show()
    plt.close()