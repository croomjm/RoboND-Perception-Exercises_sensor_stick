import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

def return_normalized_hist(features, bins_range, nbins = 32):
    hist = []
    features = np.asarray(features)
    length, depth = features.shape

    for i in range(depth):
        hist.extend(np.histogram(features[:,i], bins = nbins, range = bins_range)[0])

    hist = hist/np.sum(hist).astype(np.float)

    return hist

def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = []
    points = pc2.read_points(cloud, skip_nans=True)

    # Step through each point in the point cloud
    for point in points:
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    normed_features = return_normalized_hist(point_colors_list, bins_range = (0,256))

    #plot_histogram(normed_features, 'Color Histogram')

    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_components = pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True)

    normals = [list(n) for n in norm_components]

    normed_features = return_normalized_hist(normals, bins_range=(-1,1))

    #plot_histogram(normed_features, 'Normals Histogram')

    return normed_features

def plot_histogram(hist, title = 'Default'):
    if hist is not None:
        fig = plt.figure(figsize=(12,6))
        plt.plot(hist)
        plt.title(title, fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()
        plt.waitforbuttonpress(timeout=3)
    else:
        print('Your function is returning None...')

