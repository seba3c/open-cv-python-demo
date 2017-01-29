import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 153, 255)


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
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), COLOR_GREEN, 2)
        return True
