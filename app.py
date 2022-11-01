import os
import sys
import time
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from oakd_camera import OakDCamera


class DataCollector(QtWidgets.QMainWindow):
    def __init__(self):
        super(DataCollector, self).__init__()
        self.ui = uic.loadUi('data_collector.ui', baseinstance=self)
        self.fps = 5
        self.conf_thresh = 245
        self.save_data_dir = None
        self.save_rgb_dir = None
        self.save_disp_dir = None
        self.save_depth_dir = None
        self.folder_name = None
        self.image_count = 0
        self.image_width = 1280
        self.image_height = 720
        self.is_streaming = False
        self.is_captured = False
        self.is_origin = True
        self.setupUI()
        self.camera = OakDCamera(self.fps, self.conf_thresh)

    def setupUI(self):
        # Find Qt objects in ui file
        self.capture_btn = self.ui.findChild(QtWidgets.QPushButton, 'captureButton')
        self.browse_btn = self.ui.findChild(QtWidgets.QPushButton, 'browseFolderButton')
        self.browse_line_edit = self.ui.findChild(QtWidgets.QLineEdit, 'browseFolderLineEdit')
        self.start_stream_btn = self.ui.findChild(QtWidgets.QPushButton, 'startStreamButton')
        self.stop_stream_btn = self.ui.findChild(QtWidgets.QPushButton, 'stopStreamButton')
        self.fps_spin = self.ui.findChild(QtWidgets.QSpinBox, 'fpsSpinBox')
        self.conf_thresh_spin = self.ui.findChild(QtWidgets.QSpinBox, 'confThreshSpinBox')
        self.rgb_label = self.ui.findChild(QtWidgets.QLabel, 'rgbViewLabel')
        self.disp_label = self.ui.findChild(QtWidgets.QLabel, 'dispViewLabel')
        self.depth_label = self.ui.findChild(QtWidgets.QLabel, 'depthViewLabel')
        self.origin_checkbox = self.ui.findChild(QtWidgets.QCheckBox, 'captureOriginalCheckbox')
        #
        self.view_width = self.rgb_label.width()
        self.view_height = self.rgb_label.height()
        self.fps_spin.setValue(self.fps)
        self.conf_thresh_spin.setValue(self.conf_thresh)
        # Connect Qt objects to methods
        self.capture_btn.clicked.connect(self.capture_btn_clicked)
        self.browse_btn.clicked.connect(self.browse_btn_clicked)
        self.start_stream_btn.clicked.connect(self.start_stream_btn_clicked)
        self.stop_stream_btn.clicked.connect(self.stop_stream_btn_clicked)
        self.fps_spin.valueChanged.connect(self.fps_spin_valueChanged)
        self.conf_thresh_spin.valueChanged.connect(self.conf_thresh_spin_valueChanged)
        self.origin_checkbox.toggled.connect(self.origin_checkbox_toggled)

    def closeEvent(self, event):
        if self.is_streaming:
            msg = QtWidgets.QMessageBox(text='Please stop streaming before close!')
            msg.exec_()
            event.ignore()
            return
        self.camera.kill()
        self.camera.close()
        event.accept()

    def capture_btn_clicked(self):
        if not self.is_captured:
            if self.save_data_dir is None or self.save_data_dir == '':
                msg = QtWidgets.QMessageBox(text='The saving path is empty!')
                msg.exec_()
                return
            if not self.is_streaming:
                msg = QtWidgets.QMessageBox(text='Please stream images!')
                msg.exec_()
                return
            self.image_count = 0
            self.folder_name = time.strftime('%Y%m%d%H%M%S')
            self.save_rgb_dir = os.path.join(self.save_data_dir, 'rgb', self.folder_name)
            self.save_disp_dir = os.path.join(self.save_data_dir, 'disp', self.folder_name)
            self.save_depth_dir = os.path.join(self.save_data_dir, 'depth', self.folder_name)
            if not os.path.exists(self.save_rgb_dir):
                os.makedirs(self.save_rgb_dir)
            if not os.path.exists(self.save_disp_dir):
                os.makedirs(self.save_disp_dir)
            if not os.path.exists(self.save_depth_dir):
                os.makedirs(self.save_depth_dir)
            self.camera.signals.connect(self.save_data)
            self.is_captured = True
            self.capture_btn.setText('Stop')
            self.origin_checkbox.setEnabled(False)
        else:
            self.camera.signals.disconnect(self.save_data)
            self.is_captured = False
            self.capture_btn.setText('Capture')
            self.origin_checkbox.setEnabled(True)

    def browse_btn_clicked(self):
        self.save_data_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a directory', os.getcwd())
        if self.save_data_dir:
            self.save_data_dir = QtCore.QDir.toNativeSeparators(self.save_data_dir)
        self.browse_line_edit.setText(self.save_data_dir)

    def start_stream_btn_clicked(self):
        if not self.is_streaming:
            self.camera.resume()
            self.is_streaming = True
        self.camera.start(QtCore.QThread.Priority.LowPriority)
        self.camera.signals.connect(self.get_data)
        self.start_stream_btn.setDisabled(True)
        self.stop_stream_btn.setEnabled(True)

    def stop_stream_btn_clicked(self):
        if self.is_captured:
    	    msg = QtWidgets.QMessageBox(text='Please stop capture before stopping stream!')
    	    msg.exec_()
    	    return
        self.is_streaming = False
        self.camera.pause()
        self.camera.signals.disconnect(self.get_data)
        self.stop_stream_btn.setDisabled(True)
        self.start_stream_btn.setEnabled(True)
        self.rgb_label.clear()
        self.disp_label.clear()
        self.depth_label.clear()

    def fps_spin_valueChanged(self):
        self.fps = self.fps_spin.value()

    def conf_thresh_spin_valueChanged(self):
        self.conf_thresh = self.conf_thresh_spin.value()

    def origin_checkbox_toggled(self, checked):
        self.is_origin = checked
        if checked:
            self.image_width = 1280
            self.image_height = 720
        else:
            self.image_width = 640
            self.image_height = 360

    def get_data(self, data):
        print('view', data[2].max())
        self.update_view_label(data[0], self.rgb_label, 'rgb')
        self.update_view_label(data[1], self.disp_label, 'disp')
        self.update_view_label(data[2], self.depth_label, 'depth')

    def save_data(self, data):
        self.image_count += 1
        print('save', data[2].max())
        cv2.imwrite(os.path.join(self.save_rgb_dir, self.folder_name + str(self.image_count) + '.png'), data[0])
        cv2.imwrite(os.path.join(self.save_disp_dir, self.folder_name + str(self.image_count) + '.png'), data[1])
        cv2.imwrite(os.path.join(self.save_depth_dir, self.folder_name + str(self.image_count) + '.png'), data[2])

    def update_view_label(self, img, label, mode='rgb'):
        if mode == 'disp':
            img = (img * (255 / 96)).astype(np.uint8)
            # img = cv2.equalizeHist(img)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        elif mode == 'depth':
            img = cv2.normalize(img, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # img = cv2.equalizeHist(img)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
        img = cv2.resize(img, (self.view_width, self.view_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        img = QtGui.QPixmap(img)
        label.setPixmap(img)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = DataCollector()
    ex.show()
    sys.exit(app.exec_())
