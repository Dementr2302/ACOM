import cv2
import numpy as np


def start_point():
    # Task 1: Read the image from the camera and translate it into HSV format.
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Task 2: Apply image filtering using the command inRange
        # and leave only the red part.
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_only = cv2.bitwise_and(frame, frame, mask=mask)

        # Task 3: Perform morphological transformations of the filtered image.
        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(mask, kernel, iterations=1)
        # dilation = cv2.dilate(mask, kernel, iterations=1)

        image_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        image_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Task 4: Find the moments in the resulting image of the first order,
        # find the area of the object.
        moments = cv2.moments(mask)
        area = moments['m00']

        # Task 5: Based on the analysis of the area of the object,
        # find its center and build a black rectangle around the object.
        if area > 0:
            width = height = int(np.sqrt(area))
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])

            cv2.rectangle(
                frame,
                (c_x - (width // 16), c_y - (height // 16)),
                (c_x + (width // 16), c_y + (height // 16)),
                (0, 0, 0),
                2
            )

        # Display the resulting frames
        cv2.imshow('frame', frame)
        cv2.imshow('Red Only', red_only)
        cv2.imshow("Opening", image_opening)
        cv2.imshow("Closing", image_closing)
        # cv2.imshow('Erosion', erosion)
        # cv2.imshow('Dilation', dilation)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

