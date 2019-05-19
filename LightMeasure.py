"""required version of python 3.7
input field - field referring to video file (avi or mp4)
output filed - empty (input_name + ".csv"), file or directory
chunk size - size of square to proceed. chunk numeration direction from left to right, from top to bottom of frame"""


import pip.__main__ as pip
for module_name in "pip", "numpy", "PyQt5", "opencv-python":
    pip._main(["install", module_name])


import sys
import os
import itertools
import cv2
from PyQt5.QtWidgets import *


def sorting(value):
    return .07 * int(value[0]) + .72 * int(value[1]) + .21 * int(value[2])


def proceedVideo(square_size, video_name, result_file):
    video = cv2.VideoCapture(video_name)
    result = open(result_file, "w", encoding="utf-8")

    for f in itertools.count():
        ret, frame = video.read()
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_h, max_w, colors = frame.shape
        result.write("%i" % f)

        for w in range(0, max_w, square_size):
            for h in range(0, max_h, square_size):
                frame_piece = frame[h:h + square_size, w:w + square_size]
                if colors == 1:     # gray picture
                    frame_piece = cv2.cvtColor(frame_piece, cv2.COLOR_GRAY2BGR)

                frame_plain = frame_piece.reshape((-1, 3))
                pixel = max(frame_plain, key=sorting)
                gray = sorting(pixel)
                b, g, r = pixel
                result.write(" %i %i %i %i" % (gray, r, g, b))

        result.write("\n")


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        widget = QWidget()
        self.resize(400, 120)
        self.setCentralWidget(widget)
        layout = QGridLayout(widget)
        layout.addWidget(QLabel("input file"), 0, 0, 1, 2)
        layout.addWidget(QLabel("output file or directory"), 2, 0, 1, 2)
        layout.addWidget(QLabel("chunk size"), 4, 1, 1, 1)

        self.input_field = QLineEdit()
        layout.addWidget(self.input_field, 1, 0, 1, 1)
        self.input_button = QPushButton("<")
        self.input_button.setFixedWidth(40)
        layout.addWidget(self.input_button, 1, 1, 1, 1)

        self.output_field = QLineEdit()
        layout.addWidget(self.output_field, 3, 0, 1, 1)
        self.output_button = QPushButton("<")
        self.output_button.setFixedWidth(40)
        layout.addWidget(self.output_button, 3, 1, 1, 1)
        self.button_run = QPushButton("run")
        self.button_run.setFixedHeight(40)
        layout.addWidget(self.button_run, 5, 0, 1, 2)
        self.spin = QSpinBox()
        self.spin.setValue(4)
        self.spin.setMinimum(1)
        layout.addWidget(self.spin, 4, 0, 1, 1)

        self._menu = QMenu()

        self.output_button.setMenu(self._menu)
        self._menu.addAction("file", self._setOutFile)
        self._menu.addAction("directory", self._setOutDir)
        self.input_button.released.connect(self._setInput)
        self.button_run.released.connect(self._run)


    def _setInput(self):
        try:
            self.input_field.setText(QFileDialog().getOpenFileName()[0])
        except:
            self.input_field.setText("")

    def _setOutFile(self):
        try:
            self.output_field.setText(QFileDialog().getSaveFileName(filter="*.txt")[0])
        except:
            self.output_field.setText("")

    def _setOutDir(self):
        try:
            self.output_field.setText(QFileDialog().getExistingDirectory())
        except:
            self.output_field.setText("")

    def _run(self):
        in_file = self.input_field.text()
        if not os.path.exists(in_file):
            print("input file doesn't exists")
            return

        out_file = self.output_field.text()
        if not out_file:
            out_file = os.path.splitext(in_file)[0] + ".csv"
        elif os.path.isdir(out_file):
            out_file = os.path.join(out_file, os.path.splitext(in_file)[0] + ".csv")

        print("out file: " + out_file)
        try:
            proceedVideo(self.spin.value(), in_file, out_file)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
