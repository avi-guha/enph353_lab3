#!/usr/bin/env python3
import rospy
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


## @package line_follower
#  A ROS node for line following using image processing and PID control.
#
#  This node subscribes to a camera topic, processes images to detect a line,
#  and publishes velocity commands to follow the line. If the line is lost,
#  it attempts recovery maneuvers using the last known direction.

class LineFollower:
    ## Constructor.
    #  Initializes the ROS node, subscribers, publishers, PID parameters,
    #  and control/recovery settings.
    def __init__(self):
        rospy.init_node('line_follower')

        ## Bridge to convert between ROS Image messages and OpenCV images.
        self.bridge = cv_bridge.CvBridge()

        ## Subscriber for the raw camera image.
        self.image_sub = rospy.Subscriber('/image_raw', Image, self.image_callback)

        ## Publisher for velocity commands.
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # --- PID parameters ---
        ## Proportional gain.
        self.kp = 0.05
        ## Integral gain.
        self.ki = 0.00
        ## Derivative gain.
        self.kd = 0.02

        ## Previous error term (for derivative calculation).
        self.prev_error = 0
        ## Integral accumulator (for integral calculation).
        self.integral = 0

        # --- Control settings ---
        ## Forward speed of the robot while following the line.
        self.forward_speed = 0.5

        # --- Recovery settings ---
        ## Last known turn direction (1.0 = left, -1.0 = right).
        self.last_direction = 0.0
        ## Angular speed during recovery turns.
        self.recovery_turn_speed = 3.14
        ## Forward speed during recovery.
        self.recovery_forward_speed = 0.0

    ## Callback function for image messages.
    #  Converts the incoming image to grayscale, thresholds it, and attempts
    #  to detect the line position using two regions of interest (bottom and mid).
    #  Computes a PID-based control signal and publishes a velocity command.
    #
    #  @param msg The incoming ROS Image message.
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        height, width = thresh.shape

        # Define regions of interest (ROI)
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
            # PID error terms
            error = cx_full - width // 2
            self.integral += error
            derivative = error - self.prev_error

            # PID controller output for angular velocity
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
            # Recovery behavior when line is lost
            twist.linear.x = self.recovery_forward_speed
            twist.angular.z = self.recovery_turn_speed * self.last_direction

        # Publish twist command
        self.cmd_pub.publish(twist)

        # Debug visualization (disabled by default)
        # Uncomment for visualization in OpenCV windows
        #
        # debug_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(debug_frame,
        #               (int(0.2 * width), int(0.7 * height)),
        #               (int(0.8 * width), int(1.0 * height)),
        #               (0, 255, 0), 2)
        # cv2.rectangle(debug_frame,
        #               (int(0.0 * width), int(0.0 * height)),
        #               (int(1.0 * width), int(0.5 * height)),
        #               (255, 0, 0), 2)
        #
        # if cx_full is not None:
        #     cv2.circle(debug_frame, (cx_full, int(0.9 * height)), 5, (0, 0, 255), -1)
        #
        # cv2.imshow("Line Following Dual ROI", debug_frame)
        # cv2.waitKey(1)


if __name__ == '__main__':
    ## Main execution block.
    #  Instantiates the LineFollower class and starts the ROS event loop.
    try:
        LineFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
