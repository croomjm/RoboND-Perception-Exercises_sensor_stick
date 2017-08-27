#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import time

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

class Segmenter(object):
    def __init__(self, model):
        #assign SVM model parameters
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']

    def get_normals(self, cloud):
        get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
        return get_normals_prox(cloud).cluster

    def return_color_list(self, cluster_count, color_list = []):
        if (cluster_count > len(color_list)):
            for i in range(len(color_list), cluster_count):
                color_list.append(random_color_gen())
        return color_list

    def voxel_grid_downsample(self, cloud, leaf_size=0.01):
        vox = cloud.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
        cloud = vox.filter()

        return cloud

    def axis_passthrough_filter(self, cloud, axis, bounds):
        assert(axis in ['x','y','z'])

        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name(axis)
        axis_min, axis_max = bounds
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud = passthrough.filter()

        return cloud

    def ransac_plane_segmentation(self, cloud, max_distance = 0.01):
        seg = cloud.make_segmenter() 
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(max_distance)

        inliers, coefficients = seg.segment()

        outlier_cloud = cloud.extract(inliers, negative=True)
        inlier_cloud = cloud.extract(inliers, negative=False)

        return inlier_cloud, outlier_cloud

    def get_euclidean_cluster_indices(self, cloud, tolerance, size_bounds):
        XYZcloud = XYZRGB_to_XYZ(cloud)

        min_cluster_size, max_cluster_size = size_bounds
        tree = XYZcloud.make_kdtree()
        
        # Create a cluster extraction object
        ec = XYZcloud.make_EuclideanClusterExtraction()
        
        # Set tolerances for distance threshold 
        # as well as minimum and maximum cluster size (in points)
        ec.set_ClusterTolerance(tolerance)
        ec.set_MinClusterSize(min_cluster_size)
        ec.set_MaxClusterSize(max_cluster_size)
        
        # Search the k-d tree for clusters
        ec.set_SearchMethod(tree)
        
        # Extract indices for each of the discovered clusters
        cluster_indices = ec.Extract()

        return cluster_indices

    def return_colorized_clusters(self, cloud, cluster_indices, color_list = []):
        colorized_clusters_list = []
        cluster_colors = self.return_color_list(len(cluster_indices), color_list)

        for j, indices in enumerate(cluster_indices):
            color = rgb_to_float(cluster_colors[j])
            for i in indices:
                colorized_clusters_list.append([cloud[i][0], cloud[i][1], cloud[i][2], color])

        colorized_clusters = pcl.PointCloud_PointXYZRGB()
        colorized_clusters.from_list(colorized_clusters_list)

        return colorized_clusters

    def convert_and_publish(self, message_pairs):
        for m in message_pairs:
            cloud, publisher = m
            ros_cloud = pcl_to_ros(cloud)
            publisher.publish(ros_cloud)
            print('     Publishing {}.'.format(publisher.name))

    def detect_objects(self, cloud, cluster_indices):
        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        detected_objects = []
        positions = []

        for j, indices in enumerate(cluster_indices):
            # Grab the points for the cluster
            cluster = cloud.extract(indices)

            positions.append(list(cluster[0][:3]))

            cluster = pcl_to_ros(cluster)

            # Compute the associated feature vector
            chists = compute_color_histograms(cluster, using_hsv=True)
            normals = self.get_normals(cluster)
            nhists = compute_normal_histograms(normals)
            feature_vector = np.concatenate((chists, nhists))

            # Make the prediction
            prediction = self.clf.predict(self.scaler.transform(feature_vector.reshape(1,-1)))
            label = self.encoder.inverse_transform(prediction)[0]

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = cluster
            detected_objects.append(do)
            detected_objects_labels.append(label)

        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

        return detected_objects, detected_objects_labels, positions

    def publish_detected_objects(self, detected_objects, positions, marker_pub, objects_pub):
        for index, do in enumerate(detected_objects):
            # Publish the label into RViz
            label_pos = positions[index]
            label_pos[2] += .4 #move above the object
            marker_pub.publish(make_label(do.label,label_pos,index))

        objects_pub.publish(detected_objects)

class handle_clustering(object):
    def __init__(self, model_file):
        #load SVM object classifier
        self.model = pickle.load(open(model_file, 'rb'))

        #initialize object segmenter
        self.segmenter = Segmenter(self.model)

        # TODO: ROS node initialization
        rospy.init_node('clustering', anonymous=True)

        # TODO: Create Subscribers
        self.pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, self.pcl_callback, queue_size=1)

        # TODO: Create Publishers
        #self.reduced_cloud_pub = rospy.Publisher("/pcl_reduced", PointCloud2, queue_size=1)
        #self.objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
        #self.table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
        self.colorized_cluster_pub = rospy.Publisher("/colorized_clusters", PointCloud2, queue_size=1)
        self.object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        self.detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size = 1)

        #self.passthroughy_filter_pub = rospy.Publisher("/y_passthrough_filter", PointCloud2, queue_size=1)
        #self.passthroughz_filter_pub = rospy.Publisher("/z_passthrough_filter", PointCloud2, queue_size=1)
        #self.ransac_filter_pub = rospy.Publisher("/ransac_filter", PointCloud2, queue_size = 1)
        #self.evaluating_cluster_pub = rospy.Publisher("/evaluating_cluster", PointCloud2, queue_size = 1)

        #initialized colorized object cloud color list
        self.color_list = []

        print('PCL Publishers Initialized.')
        rospy.spin()

    def pcl_callback(self, pcl_msg):
        print('PCL Message Received!')
        seg = self.segmenter #to reduce verbosity below

                # TODO: Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        leaf_size = 0.01

        # TODO: Voxel Grid Downsampling
        print('Reducing voxel resolution.')
        cloud = seg.voxel_grid_downsample(cloud, leaf_size = leaf_size)
        decimated_cloud = cloud

        # TODO: PassThrough Filter
        print('Applying passthrough filters.')
        cloud = seg.axis_passthrough_filter(cloud, 'z', (0.6, 1.1)) #filter below table
        passthroughz_cloud = cloud
        cloud = seg.axis_passthrough_filter(cloud, 'y', (-10, -1.35)) #filter out table front edge
        #passthroughy_cloud = cloud

        # TODO: RANSAC Plane Segmentation
        # TODO: Extract inliers and outliers
        print('Performing plane segmentation.')
        table_cloud, objects_cloud = seg.ransac_plane_segmentation(cloud, max_distance = leaf_size)

        # TODO: Euclidean Clustering
        # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
        print('Finding object clusters.')
        cluster_indices = seg.get_euclidean_cluster_indices(objects_cloud, 0.03, (10,5000))
        colorized_object_clusters = seg.return_colorized_clusters(objects_cloud, cluster_indices, self.color_list)
        detected_objects, detected_objects_labels, positions = seg.detect_objects(objects_cloud, cluster_indices)

        # TODO: Convert PCL data to ROS messages
        # TODO: Publish ROS messages
        print('Converting PCL data to ROS messages.')
        message_pairs = [#(decimated_cloud, self.reduced_cloud_pub),
                         #(objects_cloud, self.objects_pub),
                         #(table_cloud, self.table_pub),
                         (colorized_object_clusters, self.colorized_cluster_pub)]
                         #(passthroughy_cloud, self.passthroughy_filter_pub),
                         #(passthroughz_cloud, self.passthroughz_filter_pub)]
        
        seg.convert_and_publish(message_pairs)

        #publish detected objects and labels
        seg.publish_detected_objects(detected_objects,
                                     positions,
                                     self.object_markers_pub,
                                     self.detected_objects_pub)

if __name__ == '__main__':

    #instantiate handle_clustering object
    h = handle_clustering('model.sav')
