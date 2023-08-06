import cv2

class LINE:
    def __init__(self, frame):
        self.image = frame
        self.Point1 = None
        self.Point2 = None
        self.click_count = 0
        self.click_points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.click_count < 2:
            self.click_points.append((x, y))
            self.click_count += 1
            cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)  # Yellow color (BGR)

        cv2.imshow('Click Points Full', self.image)


    def returnPoints(self):
        cv2.namedWindow('Click Points Full', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Click Points Full', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.setMouseCallback('Click Points Full', self.mouse_callback)

        while self.click_count < 2:
            key = cv2.waitKey(10)
            if key == 27:  # Exit loop if the 'Esc' key is pressed
                break
        cv2.destroyAllWindows()

        self.Point1 = self.click_points[0]
        self.Point2 = self.click_points[1]

        return (self.Point1,self.Point2)

