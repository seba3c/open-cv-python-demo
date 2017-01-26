import logging

import cv2
import numpy as np

from ar_demo.pose_estimator import PoseEstimator
from ar_demo.utils import ROISelector, COLOR_BLUE, COLOR_RED

logger = logging.getLogger(__name__)


WIN_NAME = 'Tracker'


class VideoHandler(object):

    def __init__(self):
        logger.debug("Initializing ROI selector...")
        self.cap = cv2.VideoCapture(0)
        self.paused = False
        self.frame = None
        self.pose_tracker = PoseEstimator()
        cv2.namedWindow(WIN_NAME)
        self.roi_selector = ROISelector(WIN_NAME, self.on_rect)

    def on_rect(self, rect):
        logger.debug("on_rect event fired! (%d, %d, %d, %d)" % rect)
        self.pose_tracker.add_target(self.frame, rect)

    def start(self):
        while True:
            is_running = not self.paused and self.roi_selector.selected_rect is None
            if is_running or self.frame is None:
                ret, frame = self.cap.read()
                scaling_factor = 0.5
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                                   interpolation=cv2.INTER_AREA)
                if not ret:
                    break
                self.frame = frame.copy()
            img = self.frame.copy()
            if is_running:
                tracked = self.pose_tracker.track_target(self.frame)
                for item in tracked:
                    cv2.polylines(img, [np.int32(item.quad)], True, COLOR_RED, 3)
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 1, COLOR_BLUE)
            self.roi_selector.draw_rect(img)
            cv2.imshow(WIN_NAME, img)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.pose_tracker.clear_targets()
            if ch == 27:
                break

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Starting plannar object tracking demo...")
    VideoHandler().start()
