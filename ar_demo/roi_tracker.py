import sys
import logging

from collections import namedtuple
import numpy as np
import cv2


logger = logging.getLogger(__name__)


BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

WIN_NAME = 'Tracker'


class PoseEstimator(object):

    def __init__(self):
        # Use locality sensitive hashing algorithm
        logger.debug("Initializing Pose Estimator...")
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad')
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    # Function to add a new target for tracking
    def add_target(self, image, rect, data=None):
        logger.debug("Adding new target (%d,%d,%d,%d)...", *rect)
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []
        for keypoint, descriptor in zip(*self.detect_features(image)):
            x, y = keypoint.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(keypoint)
                descriptors.append(descriptor)
        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints,
                                 descriptors=descriptors, data=data)
        self.tracking_targets.append(target)
        logger.debug("Target added!")

    # To get a list of detected objects
    def track_target(self, frame):
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame)
        if len(self.cur_keypoints) < self.min_matches:
            return []
        matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
        matches = [match[0] for match in matches if len(match) == 2 and match[0].distance < match[1].distance * 0.75]
        if len(matches) < self.min_matches:
            return []
        matches_using_index = [[] for _ in range(len(self.tracking_targets))]
        for match in matches:
            matches_using_index[match.imgIdx].append(match)

        tracked = []
        for image_index, matches in enumerate(matches_using_index):
            if len(matches) < self.min_matches:
                continue
            target = self.tracking_targets[image_index]
            points_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in matches]
            points_prev, points_cur = np.float32((points_prev, points_cur))
            H, status = cv2.findHomography(points_prev, points_cur, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < self.min_matches:
                continue
            points_prev, points_cur = points_prev[status], points_cur[status]
            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([[x_start, y_start], [x_end, y_start], [x_end, y_end],
                               [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            track = self.tracked_target(target=target, points_prev=points_prev,
                                        points_cur=points_cur, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key=lambda x: len(x.points_prev), reverse=True)
        return tracked

    # Detect features in the selected ROIs and return the keypoints and descriptors
    def detect_features(self, frame):
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    # Function to clear all the existing targets
    def clear_targets(self):
        logger.debug("Clearing targets...")
        self.feature_matcher.clear()
        self.tracking_targets = []
        logger.debug("Targets cleared!")


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
                    cv2.polylines(img, [np.int32(item.quad)], True, RED, 3)
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 1, BLUE)
            self.roi_selector.draw_rect(img)
            cv2.imshow(WIN_NAME, img)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.pose_tracker.clear_targets()
            if ch == 27:
                break


class ROISelector(object):

    def __init__(self, win_name, callback_func):
        logger.debug("Initializing ROI selector...")
        self.win_name = win_name
        self.callback_func = callback_func
        cv2.setMouseCallback(self.win_name, self.on_mouse_event)
        self.selection_start = None
        self.selected_rect = None

    def on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug("mouse event fired! 'mouse left button down' (%d,%d)", x, y)
            self.selection_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.selection_start:
            logger.debug("mouse event fired! 'mouse left button up' (%d,%d)", x, y)
            logger.debug("calculating rect...")
            x_orig, y_orig = self.selection_start
            x_start, y_start = np.minimum([x_orig, y_orig], [x, y])
            x_end, y_end = np.maximum([x_orig, y_orig], [x, y])
            self.selected_rect = None
            if x_end > x_start and y_end > y_start:
                self.selected_rect = (x_start, y_start, x_end, y_end)
                logger.debug("New rect selected (%d,%d,%d,%d)", *self.selected_rect)
            self.selection_start = None
            if self.selected_rect:
                self.callback_func(self.selected_rect)

    def draw_rect(self, img):
        if not self.selected_rect:
            return False
        x_start, y_start, x_end, y_end = self.selected_rect
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), GREEN, 2)
        return True

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    logger.debug("Starting ROI tracker...")
    VideoHandler().start()
