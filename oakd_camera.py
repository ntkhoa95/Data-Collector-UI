import time
import numpy as np
from PyQt5 import QtCore
import depthai as dai
import cv2
import math
from hf import fill_bg_with_fg
from scipy.ndimage.filters import median_filter
from scipy.signal import medfilt2d
from skimage.filters import median

class wlsFilter:
    wlsStream = "wlsFilter"

    def on_trackbar_change_lambda(self, value):
        self._lambda = value * 100

    def on_trackbar_change_sigma(self, value):
        self._sigma = value / float(10)

    def __init__(self, _lambda, _sigma):
        self._lambda = _lambda
        self._sigma = _sigma
        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

    def filter(self, disparity, right, depthScaleFactor):
        filteredDisp = disparity
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
        # self.wlsFilter.setLambda(self._lambda)
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
        self.wlsFilter.setSigmaColor(self._sigma)
        # filteredDisp = self.wlsFilter.filter(disparity, right)

        # Compute depth from disparity (32 levels)
        # filteredDisp[filteredDisp == 0] = 0.00001
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            # raw depth values
            depthFrame = (depthScaleFactor / filteredDisp).astype(np.uint16)


        return filteredDisp, depthFrame


class OakDCamera(QtCore.QThread):
    signals = QtCore.pyqtSignal(list)

    def __init__(self, fps, conf_thresh):
        super(OakDCamera, self).__init__()
        self.fps = fps
        self.conf_thresh = conf_thresh
        self.rgb_img = None
        self.depth_img = None
        self.disp_img = None
        self.rectified_right_img = None
        self.is_killed = False
        self.is_paused = False
        self.wlsFilter = wlsFilter(_lambda=8000, _sigma=1.)

        self.baseline = 75  # mm
        self.disp_levels = 96
        self.fov = 71.86

        self.mutex = QtCore.QMutex()
        self._init_camera()

    def _init_camera(self):
        # Define Oak-D pipeline
        pipeline = dai.Pipeline()
        # Define RGB source
        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb_cam.initialControl.setManualFocus(135)
        rgb_cam.setIspScale(2, 3)
        rgb_cam.setFps(self.fps)
        # Define mono sources
        # Define left source
        left_cam = pipeline.createMonoCamera()
        left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        left_cam.setFps(self.fps)
        # Define right source
        right_cam = pipeline.createMonoCamera()
        right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        right_cam.setFps(self.fps)
        # Define stereo source
        stereo_depth = pipeline.createStereoDepth()
        stereo_depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo_depth.setRectifyEdgeFillColor(0)
        stereo_depth.setConfidenceThreshold(self.conf_thresh)
        stereo_depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
        stereo_depth.setLeftRightCheck(True)
        # Define outputs
        rgb_out = pipeline.createXLinkOut()
        rgb_out.setStreamName('rgb')
        disp_out = pipeline.createXLinkOut()
        disp_out.setStreamName('disparity')
        depth_out = pipeline.createXLinkOut()
        depth_out.setStreamName('depth')
        # Link sources to outputs
        rgb_cam.isp.link(rgb_out.input)
        left_cam.out.link(stereo_depth.left)
        right_cam.out.link(stereo_depth.right)
        stereo_depth.disparity.link(disp_out.input)
        stereo_depth.depth.link(depth_out.input)
        self.device = dai.Device(pipeline)

    def run(self):
        self.device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
        self.device.getOutputQueue(name='disparity', maxSize=4, blocking=False)
        self.device.getOutputQueue(name='depth', maxSize=4, blocking=False)
        while True:
            #self.mutex.lock()
            if self.is_killed:
                break
            #self.mutex.unlock()
            rgb_packets = self.device.getOutputQueue('rgb').tryGetAll()
            disp_packets = self.device.getOutputQueue('disparity').tryGetAll()
            depth_packets = self.device.getOutputQueue('depth').tryGetAll()
            if len(rgb_packets) > 0:
                self.rgb_img = rgb_packets[-1].getCvFrame()
            if len(disp_packets) > 0:
                self.disp_img = disp_packets[-1].getFrame()
                #indices = np.argwhere(self.disp_img == 0)
                #self.disp_img = median_filter(self.disp_img, 3, mode='constant').astype(np.int32)
                self.disp_img = fill_bg_with_fg(self.disp_img.astype(np.int32)).astype(np.uint8)

                # focal = self.disp_img.shape[1] / (2. * math.tan(math.radians(self.fov / 2)))
                # depthScaleFactor = self.baseline * focal
                # with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
                # #     raw depth values
                #     self.depth_img = (depthScaleFactor / self.disp_img.astype(np.float32)).astype(np.uint16)

            if len(depth_packets) > 0:
                self.depth_img = depth_packets[-1].getFrame()#.astype(np.float32)
                #self.depth_img = median_filter(self.depth_img, 3, mode='constant').astype(np.int32)
                self.depth_img = fill_bg_with_fg(self.depth_img.astype(np.int32)).astype(np.uint16)
                self.depth_img[self.depth_img >= 10000] = 0
            self.mutex.lock()
            if self.rgb_img is not None and self.disp_img is not None and self.depth_img is not None:
                self.signals.emit([self.rgb_img, self.disp_img, self.depth_img])
            self.mutex.unlock()
            while self.is_paused:
                #self.mutex.lock()
                if self.is_killed:
                    break
                #self.mutex.unlock()
                time.sleep(0)
            time.sleep(self.fps / 50.)

    def close(self):
        self.mutex.lock()
        self.device.close()
        self.mutex.unlock()
        
    def is_connected(self):
        return self.connected

    def kill(self):
        self.mutex.lock()
        self.is_killed = True
        self.mutex.unlock()

    def pause(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()

    def is_streaming(self):
        return not self.is_paused
