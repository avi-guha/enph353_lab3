#!/usr/bin/env python3
import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower')

        self.bridge = cv_bridge.CvBridge()

        # Subscriber for camera and publisher for cmd_vel
        self.image_sub = rospy.Subscriber('/image_raw', Image, self.image_callback)

        
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # PID parameters
        self.kp = 0.05   
        self.ki = 0.00   
        self.kd = 0.02   

        
        self.prev_error = 0
        self.integral = 0

        #control
        self.forward_speed = 0.5  

        # Recovery settings
        self.last_direction = 0.0      
        self.recovery_turn_speed = 3.14 
        self.recovery_forward_speed = 0.0  

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        height, width = thresh.shape

        roi_bottom = thresh[int(0.7 * height):int(1.0 * height), int(0.2 * width):int(0.8 * width)]
        roi_mid = thresh[int(0.0 * height):int(0.5 * height), int(0.0 * width):int(1.0 * width)]

        twist = Twist()
        cx_full, found = None, False

        # Try bottom ROI first
        M = cv2.moments(roi_bottom)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cx_full = cx + int(0.33 * width)
            found = True
        else:
            # Fallback to mid ROI
            M = cv2.moments(roi_mid)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cx_full = cx + int(0.33 * width)
                found = True

        if found:
            error = cx_full - width // 2
            self.integral += error
            derivative = error - self.prev_error

            angular_z = -(self.kp * error +
                          self.ki * self.integral +
                          self.kd * derivative)

            self.prev_error = error

            # Update last direction for recovery
            if angular_z > 0:
                self.last_direction = 1.0
            elif angular_z < 0:
                self.last_direction = -1.0

            twist.linear.x = self.forward_speed
            twist.angular.z = angular_z

        else:
            twist.linear.x = self.recovery_forward_speed
            twist.angular.z = self.recovery_turn_speed * self.last_direction

        # Publish twist command
        self.cmd_pub.publish(twist)

        # Debug visualization
        # debug_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(debug_frame,
        #               (int(0.2 * width), int(0.7 * height)),
        #               (int(0.8 * width), int(1.0 * height)),
        #               (0, 255, 0), 2)
        # cv2.rectangle(debug_frame,
        #               (int(0.0 * width), int(0.0 * height)),
        #               (int(1.0 * width), int(0.5 * height)),
        #               (255, 0, 0), 2)

        # if cx_full is not None:
        #     cv2.circle(debug_frame, (cx_full, int(0.9 * height)), 5, (0, 0, 255), -1)

        # cv2.imshow("Line Following Dual ROI", debug_frame)
        # cv2.waitKey(1)


if __name__ == '__main__':
    try:
        LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




