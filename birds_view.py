import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import random
from random import *
import ros_numpy
from PIL import Image, ImageEnhance
import pcl
import pcl_helper
import math
class Birdeye():
    def __init__(self):
        self._pub = rospy.Publisher('result', PointCloud2, queue_size=1)
        self.count = 0
        self.rgb_map = 0.1
    def scale_to_255(self,a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)


    def hight_from_pixel(self,pixel, min, max):
        return (pixel*(max-min)/255)+min


    def in_range_points(self,points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]

    def velo_points_2_top_view(self, points, x_range, y_range, z_range, scale):

        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)

        # extract in-range points
        x_lim = self.in_range_points(x, x, y, z, x_range, y_range, z_range)
        y_lim = self.in_range_points(y, x, y, z, x_range, y_range, z_range)

        # * x,y,z range are based on lidar coordinates
        x_size = int((y_range[1] - y_range[0])) #20
        y_size = int((x_range[1] - x_range[0])) #20

        # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
        # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        # scale - for high resolution


        x_img = -(y_lim * scale).astype(np.int32)
        y_img = -(x_lim * scale).astype(np.int32)



        # shift negative points to positive points (shift minimum value to 0)
        xx_img = x_img - int(np.floor(y_range[0]*scale))
        yy_img = y_img - int(np.floor(x_range[0]*scale))

        pixel_values = self.in_range_points(z, x, y, z, x_range, y_range, z_range)
        pixel_values  = self.scale_to_255(pixel_values, min=-10, max=10)

        # array to img
        img = np.zeros([y_size * scale , x_size * scale ], dtype=np.float32) #1126400

        img[yy_img, xx_img] = pixel_values


        center_obj = [256,137]
        car_high = img[256,137]  # pixel format
        real_hgiht_of_car = self.hight_from_pixel(car_high, -10,10)

        xx_img = 256
        yy_img = 137

        back_x_img = xx_img+int(np.floor(y_range[0]*scale))
        back_y_img = yy_img+int(np.floor(x_range[0]*scale))

        # back to lidar coordinates
        x_in_lidar = -back_x_img/scale
        y_in_lidar = -back_y_img/scale
        z_in_lidar  =  img[256,137]

        # get the distance from the detected obkject in the lidar frame

        d = math.sqrt(x_in_lidar**2 + y_in_lidar**2)
        print(x_in_lidar, y_in_lidar, z_in_lidar,d )

        return img
    def callback(self, msg):
        scan = ros_numpy.numpify(msg)
        velo_points = np.zeros((scan.shape[-1],4))
        velo_points[:,0] = scan['x']
        velo_points[:,1] = scan['y']
        velo_points[:,2] = scan['z']
        #velo_points[:,:,3] = scan['intensity']


        # Plot result
        top_image = self.velo_points_2_top_view(velo_points, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), scale=20)

        #top_image = Image.fromarray(top_image)
        save_dir = '/home/walid/walid_imad/Training-dataset/'
        #top_image = cv2.applyColorMap(top_image, cv2.COLORMAP_JET)
        cv2.imwrite('/home/walid/walid_imad/Training-dataset/{}.png'.format(self.count), top_image)
        #top_image.save(save_dir + "top_image%d.jpg" % self.count)
        self.count = self.count + 1
        cv2.imshow('os',top_image)
        cv2.waitKey(1)
    def main(self):
        rospy.Subscriber('/3Dlidar16_scan', PointCloud2, self.callback, queue_size=1)
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('bird_conv')
    tensor = Birdeye()
    tensor.main()
