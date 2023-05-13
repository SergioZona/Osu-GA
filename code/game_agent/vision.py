import cv2 as cv
import numpy as np
from cv2 import cuda

class Vision:

    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None
    gpu = False

    # constructor
    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        # load the image we're trying to match
        # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
        self.needle_img = cv.imread(needle_img_path, cv.COLOR_BGR2GRAY)

        # Save the dimensions of the needle image
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        # There are 6 methods to choose from:
        # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
        self.method = method

    def detect_slide_notes(self, haystack_img):
        # Convert image to grayscale
        gray_img = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary image
        _, thresh_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
        
        # Find contours in binary image
        contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        slider_contours = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > 10 and h > 10 and w < 100 and h < 100:
                slider_contours.append(contour)
        
        # Analyze contours to detect slider notes
        slide_notes = []
        # Create blank image to draw contours on for visualization
        vis_img = np.zeros(haystack_img.shape, dtype=np.uint8)
        for contour in slider_contours:
            # Check if contour is rectangular
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                # Check if contour has smooth texture
                x, y, w, h = cv.boundingRect(contour)
                roi = gray_img[y:y+h, x:x+w]
                laplacian = cv.Laplacian(roi, cv.CV_64F)
                var = np.var(laplacian)
                if var < 500:
                    # Contour is a slider note
                    slide_notes.append(approx)
                    # Draw contour on blank image for visualization
                    cv.drawContours(vis_img, [approx], -1, (0, 0, 255), 2)
        
        # Paint detected slider notes on input image
        cv.drawContours(haystack_img, slide_notes, -1, (0, 0, 255), 2)
        
        return slide_notes, haystack_img, vis_img
        
    def detect_circles(self, haystack_img):
        # This code does not work with GPU.
        # The implementation is with CPU: https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
        gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)            
        gray = cv.medianBlur(gray, 5)
        
        rows = gray.shape[0]

        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=100, param2=100,
                                minRadius=25, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(haystack_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(haystack_img, center, radius, (255, 0, 255), 3)
            
        return circles
    
    def detect_external_circles(self, haystack_img):
        # This code does not work with GPU.
        # The implementation is with CPU: https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
        gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)            
        gray = cv.medianBlur(gray, 5)
        
        rows = gray.shape[0]

        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=100, param2=100,
                                minRadius=55, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(haystack_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(haystack_img, center, radius, (20, 9, 179), 3)
            
        return circles

    def find(self, haystack_img, threshold=0.5, debug_mode=None, gpu=False):
        result = None

        if gpu:
            gsrc = cv.cuda_GpuMat()
            gtmpl = cv.cuda_GpuMat()
            result = cv.cuda_GpuMat()

            gsrc.upload(haystack_img)
            gtmpl.upload(self.needle_img)

            matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, cv.TM_CCOEFF_NORMED)
            gresult = matcher.match(gsrc, gtmpl)

            result = gresult.download()

            #min_valg, max_valg, min_locg, max_locg = cv.minMaxLoc(resultg)

        else:
            result = cv.matchTemplate(haystack_img, self.needle_img, self.method)
        
        # run the OpenCV algorithm
        #print(result)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))
        #print(locations)

        # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
        # locations by using groupRectangles().
        # First we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)
        # Apply group rectangles.
        # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
        # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
        # in the result. I've set eps to 0.5, which is:
        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        #print(rectangles)       

        points = []
        if len(rectangles):
            #print('Found needle.')

            line_color = (0, 255, 0)
            line_type = cv.LINE_4
            marker_color = (255, 0, 255)
            marker_type = cv.MARKER_CROSS

            # Loop over all the rectangles
            for (x, y, w, h) in rectangles:

                # Determine the center position
                center_x = x + int(w/2)
                center_y = y + int(h/2)
                # Save the points
                points.append((center_x, center_y))

                if debug_mode == 'rectangle':
                    # Determine the box position
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    # Draw the box
                    cv.rectangle(haystack_img, top_left, bottom_right, color=line_color, 
                                lineType=line_type, thickness=2)
                elif debug_mode == 'cross':
                    # Draw the center point
                    cv.drawMarker(haystack_img, (center_x, center_y), 
                                color=marker_color, markerType=marker_type, 
                                markerSize=40, thickness=2)
                elif debug_mode == 'point':
                    #Draw the center point
                    cv.drawMarker(haystack_img, (center_x, center_y), 
                                color=marker_color, markerType=marker_type, 
                                markerSize=2, thickness=2)

        if debug_mode:
            cv.imshow('Matches', haystack_img)
            #cv.waitKey()
            #cv.imwrite('result_click_point.jpg', haystack_img)

        return points