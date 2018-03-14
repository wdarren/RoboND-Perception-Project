# PR2 Robot 3D Perception Project

## 1. Summary
### Goal
The goal of this project is take the point cloud from the PR2's camera as input, and then through a perception pipeline and machine learning process, recognize the objects and output the object name and positions for further handling actions.
### Output

In `./yaml` folder, `output_1.yaml`, `output_2.yaml`, abd `output_2.yaml`, for objects in `test1.world`, `test2.world`, and `test3.world` respectively, in the format of the following

```
- arm_name: right
  object_name: biscuits
  pick_pose:
    orientation:
      w: 0.0
      x: 0.0
      y: 0.0
      z: 0.0
    position:
      x: 0.5751878619194031
      y: -0.0005115000531077385
      z: 0.6227399110794067
  place_pose:
    orientation:
      w: 0.0
      x: 0.0
      y: 0.0
      z: 0.0
    position:
      x: 0.0
      y: -0.71
      z: 0.605
  test_scene_num: 1
```

### Recognition Results
-   100% (3/3) objects recognized in test1.world
-   100% (5/5) objects recognized in test2.world
-   100% (8/8) objects recognized in test3.world

**test1**
![](https://i.imgur.com/Dj19ckg.png)

**test2**
![](https://i.imgur.com/EZO1B8y.png)

**test3**
![](https://i.imgur.com/oIk1Myv.png)

## 2. Perception Pipeline
* ### StatisticalOutlierRemoval Filter
There are noises in the images taken by camera, apply StatisticalOutlierRemoval filter to filter noises out.

```python
    outlier_filter = pcl_data.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Set threshold scale factor (number of std_dev)
    x = 0.1

    # Any point with a mean diatance larger than global (mean + x * std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    
    # Finally call the filter function
    pcl_data = outlier_filter.filter()
```
**Original image**
![](https://i.imgur.com/HV3jbPv.png)

**Result**
![](https://i.imgur.com/Ua0LvH4.png)

### Tansform image into voxel grid
```python
    vox = pcl_data.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    pcl_data = vox.filter()
```
**Result**
![](https://i.imgur.com/H5aTuCH.png)

* ### PassThrough filter
With Pass Through filter, we can remove the region we are not interested in, for example, region below the table, and far left and far right of the table. 

We have to apply the filter along two axis.
```python
    passthrough_z = pcl_data.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 0.77
    passthrough_z.set_filter_limits(axis_min, axis_max)

    pcl_data = passthrough_z.filter()

    passthrough = pcl_data.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)

    pcl_data = passthrough.filter()
```
**Before pass-through filter**
![](https://i.imgur.com/cGPH740.png)
**After pass-through filter**
![](https://i.imgur.com/VQH758J.png)

### RANSAC Plane Segmentation
By Random Sample Consensus (RANSAC), We do plane fitting to segment points that correspond to the table. After extracting points of table, the left points (outliers, or negative to inliers) are the points of obejects we want.
```python
    # TODO: RANSAC Plane Segmentation
    seg = pcl_data.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.02
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table = pcl_data.extract(inliers, negative=False)
    cloud_objects = pcl_data.extract(inliers, negative=True)
```
![](https://i.imgur.com/YAwmfQd.png)

### Clustering Segmentation
Now we have points of all the objects we want. To seperate them into individual objects, we have to apply Euclidean Clustering. 

```python
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()

    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(10000)

    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract()
```
## 3. Object Recognition
There are several features we can use for object recognition. Here, we use two most common features, color and shape to accomplish the task. For color, we use HSV histograms and for shape, we surface normal of the object to distinguish an object from another further.


```python
def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # TODO: Compute histograms
    h_hist = np.histogram(channel_1_vals, bins=32, range=(0,256))
    s_hist = np.histogram(channel_2_vals, bins=32, range=(0,256))
    v_hist = np.histogram(channel_3_vals, bins=32, range=(0,256))


    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    # normed_features = np.random.random(96) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)
    h_hist = np.histogram(norm_x_vals, bins=32, range=(0,256))
    s_hist = np.histogram(norm_y_vals, bins=32, range=(0,256))
    v_hist = np.histogram(norm_z_vals, bins=32, range=(0,256))
    
    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    return normed_features
```
### SVM Image Classfication

We first capture features as the training set for our classifier. 

`for` loop in `capture_features.py` that begins with `for i in range(5):` represent the number of times we capture features for each object. We use 20 here.

And then, change the flag to `using_hsv=True` to call `compute_color_histograms()` for using HSV color space.

After capturing all the features of objects, use them to train the SVM classifier

![](https://i.imgur.com/1zstw3t.png)

Feed segmented object information into the models, we can have the objects recognized.
```python
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        ros_pcl_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_pcl_cluster, using_hsv=True)
        normals = get_normals(ros_pcl_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction  = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz

        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster 
        detected_objects.append(do)
```
## 4. Output .yaml files

### Get the information

There are five variables to put into .yamlfiles

* test_scene_num: the number of test  world
* object_name: the name of object
* arm_name: based on the group, `red` group uses `left` arm, while `green` group uses `right` arm
```
dropbox:
  - name: left
    group: red
    position: [0,0.71,0.605]
  - name: right
    group: green
    position: [0,-0.71,0.605]
```

* pick_pose: use the mean of points of objects
* place_pose: position of dropbox

Codes as follows

```python
    # TODO: Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # TODO: Get/Read parameters
    objects = rospy.get_param('/object_list')
    dropboxs = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    test_scene_num.data = 1

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    yaml_dists = []
    for detected_object in object_list:

        object_name.data = str(detected_object.label)

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points = ros_to_pcl(detected_object.cloud).to_array()
        x, y, z = np.mean(points, axis = 0)[:3]
        pick_pose.position.x = np.asscalar(x) 
        pick_pose.position.y = np.asscalar(y)
        pick_pose.position.z = np.asscalar(z)

        # TODO: Create 'place_pose' for the object
        for item in objects:
            if item['name'] == object_name.data:
                for box in dropboxs:
                    if box['group'] == item['group']:
                        x, y, z = box['position']
                        place_pose.position.x = np.float(x) 
                        place_pose.position.y = np.float(y)
                        place_pose.position.z = np.float(z)        
                        arm_name.data = box['name']
```
## 5. Improvement
* PR2 Collision Avoidance
* Perform pick and place operation
