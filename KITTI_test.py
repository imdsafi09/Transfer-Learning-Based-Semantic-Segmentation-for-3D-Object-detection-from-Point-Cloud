import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import random
from random import *
import ros_numpy
from PIL import Image, ImageEnhance
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy

import math
PACKAGE_NAME = 'rviz'
import roslib; roslib.load_manifest(PACKAGE_NAME)
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA
class Birdeye():
    def get_contour_angle(self,cnt):
        """
        Return orientation of a contour
        :param cnt: contour
        :return: Angle in radians
        """
        rotrect = cv2.minAreaRect(cnt)
        angle = rotrect[-1]
        size1, size2 = rotrect[1][0], rotrect[1][1]
        ratio_size = float(size1) / float(size2)
        if 1.25 > ratio_size > 0.75:
            if angle < -45:
                angle = 90 + angle
        else:
            if size1 < size2:
                angle = angle + 180
            else:
                angle = angle + 90

            if angle > 90:
                angle = angle - 180

        return math.radians(angle)


    def __init__(self):
        self._pub = rospy.Publisher('result', PointCloud2, queue_size=1)
        self.count = 0
        self.marker_pub = rospy.Publisher('marker_test', Marker)
        self.marker_pub2 = rospy.Publisher('marker_text', Marker)
        self.rgb_map = 0.1
        self.model_pretrained = tf.keras.models.load_model('model_new.h5',custom_objects={'masked_loss': self.masked_loss})
    def normalize_depth(self, val, min_v, max_v):
        """
        print 'normalized depth value'
        normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

    def in_range_points(self,points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]



    def make_marker(self, marker_type, scale, r, g, b, a,xpos,ypos,id,angle):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.action = Marker.ADD
        m.lifetime = rospy.Time(0.5)
        m.header.frame_id = '/velo_link'

        m.header.stamp = rospy.Time.now()
        m.ns = 'marker_test_%d' % marker_type
        m.id = id
        m.type = marker_type

        # convert to Quaternion
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.w = angle
        m.pose.orientation.z = 0            #angle

        m.pose.position.x = xpos
        m.pose.position.y = ypos
        m.pose.position.z = -1.2
        m.scale = scale
        m.color.r = 0.2;
        m.color.g = 0.4;
        m.color.b = 0.6;
        m.color.a = 1.0;


        mm = Marker()
        mm.action = Marker.ADD
        mm.lifetime = rospy.Time(0.5)
        mm.header.frame_id = '/velo_link'

        mm.header.stamp = rospy.Time.now()
        mm.id = id
        mm.type = Marker.TEXT_VIEW_FACING
        mm.text='CAR %d'%id
        # convert to Quaternion
        mm.pose.orientation.x = 0
        mm.pose.orientation.y = 0
        mm.pose.orientation.w = 1
        mm.pose.orientation.z =  0

        mm.pose.position.x = xpos
        mm.pose.position.y = ypos
        mm.pose.position.z = 0.5
        mm.scale = scale
        mm.color.r = 1;
        mm.color.g = 1;
        mm.color.b = 1;
        mm.color.a = 1.0;



        return m,mm
    def velo_points_2_top_view(self, points, x_range, y_range, z_range, scale):

        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)

        # extract in-range points
        x_lim = self.in_range_points(x, x, y, z, x_range, y_range, z_range)

        y_lim = self.in_range_points(y, x, y, z, x_range, y_range, z_range)
        dist_lim = self.in_range_points(dist, x, y, z, x_range, y_range, z_range)

        # * x,y,z range are based on lidar coordinates
        x_size = int((y_range[1] - y_range[0])) #20
        y_size = int((x_range[1] - x_range[0])) #40

        # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
        # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        # scale - for high resolution
        x_img = -(y_lim * scale).astype(np.int32)
        y_img = -(x_lim * scale).astype(np.int32)

        # shift negative points to positive points (shift minimum value to 0)
        x_img += int(np.trunc(y_range[1] * scale))
        y_img += int(np.trunc(x_range[1] * scale))

        # normalize distance value & convert to depth map
        max_dist = np.sqrt((max(x_range)**2) + (max(y_range)**2))
        dist_lim = self.normalize_depth(dist_lim, min_v=0, max_v=max_dist)

        # array to img
        img = np.zeros([y_size * scale , x_size * scale], dtype=np.uint8)  #200,400
        img[y_img, x_img] = dist_lim
        print(np.shape(img))

        return img

    def masked_loss(self,y_true, y_pred):
        gt_validity_mask = tf.cast(tf.greater_equal(y_true[:, :, :, 0], 0), dtype=tf.float32)
        y_true = K.abs(y_true)
        raw_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        masked = gt_validity_mask * raw_loss
        return tf.reduce_mean(masked)
    def callback(self, msg):
        scan = ros_numpy.numpify(msg)

        velo_points = np.zeros((scan.shape[-1],4))
        velo_points[:,0] = scan['x']
        velo_points[:,1] = scan['y']
        velo_points[:,2] = scan['z']
        #velo_points[:,:,3] = scan['intensity']


        # Plot result
        top_image = self.velo_points_2_top_view(velo_points, x_range=(-10,10), y_range=(-10, 10), z_range=(-2, 0.2), scale=20)
        top_image = cv2.applyColorMap(top_image, cv2.COLORMAP_HOT)  #1.TURBO, 2.MAGMA 3.HOT
        
        img = np.flip(top_image, axis=2).astype(np.float32) / 255.
        img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        #gt = np.load(os.path.join(source_mask, files_val[i] + '.npy'))

        ret = self.model_pretrained.predict(np.expand_dims(img, axis=0))
        ret1= cv2.resize(ret[0], (400, 400), cv2.INTER_LINEAR)
        img = cv2.resize(img, (400,400), cv2.INTER_LINEAR)
        #img1 = cv2.resize(img, (401, 401), cv2.INTER_LINEAR)
        original = ret1.copy()

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        blurred = (blurred*255).astype(np.uint8)
        canny = cv2.Canny(blurred, 0, 200, 0)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Iterate thorugh contours and filter for ROI
        image_number = 0
        id = 0
        for c in cnts:
            # Obtain bounding box coordinates and draw rectangle
            rect = cv2.minAreaRect(c)
            angle = self.get_contour_angle(c)

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            image = cv2.drawContours(img,[box],-1,(0,0,150),1)
            leftmost = tuple(c[c[:,:,0].argmin()][0])
            rightmost = tuple(c[c[:,:,0].argmax()][0])
            topmost = tuple(c[c[:,:,1].argmin()][0])
            bottommost = tuple(c[c[:,:,1].argmax()][0])
            #angle = math.atan2([box][0][2][1] - [box][0][1][1], [box][0][2][0] - [box][0][1][0])
            #angle = abs(angle)-1.5708
            #angle = math.atan2(rightmost[1] - leftmost[1], rightmost[0] - leftmost[0])
            #print([box])
            #print("leftmost:{0}, rightmost:{1}, topmost:{2}, bottommost:{3}".format(leftmost,rightmost, topmost, bottommost))
            #print(angle)
            # Find center coordinate and draw center point
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img, (cx, cy), 2, (36,255,12), -1)
            cv2.putText(img,"car", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,100))
            print('X,Y_image: ({}, {})'.format(cx,cy))
            scale = 20
            xx_img = cx
            yy_img = cy
            pixel = ret1[cx,cy]
            #print(pixel)

            real_hig = (pixel*(10-(-10)/255)+(-10))+ 1.8

            id = id + 1

            back_x_img = xx_img+int(np.floor(-10*scale))
            back_y_img = yy_img+int(np.floor(-10*scale))

            # back to lidar coordinates
            x_in_lidar = -back_x_img/scale

            y_in_lidar = -back_y_img/scale
            height = real_hig

            # get the distance from the detected obkject in the lidar frame

            d = math.sqrt(x_in_lidar**2 + y_in_lidar**2)


            scale = Vector3(3.2,1.8,0.8)
            #marker_pub.publish(make_marker(Marker.SPHERE,   scale, 1, .5, .2, .3))
            #marker_pub.publish(make_marker(Marker.CYLINDER, scale, .5, .2, 1, .3))
            m,nn=self.make_marker(Marker.CUBE, scale,leftmost, rightmost, topmost, bottommost,y_in_lidar, x_in_lidar ,id,angle )
            self.marker_pub.publish(m)
            self.marker_pub2.publish(nn)

            print('X,Y,height_Lidar,Distance : ({}, {},{},{})'.format(x_in_lidar,y_in_lidar,height,d))

        cv2.imshow('image', img)
        cv2.imshow('seg', ret1)
        cv2.waitKey(1)

    def main(self):
        rospy.Subscriber('/kitti/velo/pointcloud', PointCloud2, self.callback, queue_size=1)

        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('bird_conv')
    tensor = Birdeye()
    tensor.main()
