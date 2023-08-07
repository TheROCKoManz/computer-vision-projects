import cv2

class Polygon:
    def __init__(self, frame):
        self.image = frame
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

        while self.click_count < 10:
            key = cv2.waitKey(10)
            if key == 27:  # Exit loop if the 'Esc' key is pressed
                break
        cv2.destroyAllWindows()

        return self.click_points

