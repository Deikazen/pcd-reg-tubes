import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)

        self.image = None
        self.processed_image = None

        # hubungkan tombol
        self.actionMedian.triggered.connect(self.median)
        self.loadButton.clicked.connect(self.loadClicked)

        # Operasi Titik
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionContrast.triggered.connect(self.contrast)
        self.actionNegative.triggered.connect(self.negative)
        self.actionThreshold.triggered.connect(self.threshold)

    @pyqtSlot()
    def loadClicked(self):
        # Membuka dialog pencarian file
        # Argumen: (parent, judul, direktori awal, filter format file)
        options = QFileDialog.Options()
        flname, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp)",
                                                options=options)

        # Cek jika user tidak membatalkan pilihan (flname tidak kosong)
        if flname:
            self.loadImage(flname)

    def loadImage(self, flname):
        self.image = cv2.imread(flname)
        if self.image is not None:
            self.displayImage(self.image, self.imgLabel)
        else:
            print(f"Error: File {flname} tidak ditemukan.")

    @pyqtSlot()
    def grayClicked(self):
        if self.image is None:
            return

    def median(self):
        if self.image is None:
            return

        # 1. Konversi citra ke grayscale
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 2. Copy image (sebagai kanvas output)
        img_out = img_gray.copy()

        # 3. h = ukuran tinggi citra, 4. w = ukuran baris citra
        h, w = img_gray.shape[:2]

        # 5. Looping piksel sesuai algoritma
        # Menggunakan range 3 sampai h-3 dan w-3 untuk menghindari error out-of-bounds di tepi gambar
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []

                # Mengambil nilai tetangga dalam kernel 7x7
                # Di Python, range(-3, 4) berarti memutar nilai: -3, -2, -1, 0, 1, 2, 3
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = img_gray[i + k, j + l]
                        neighbors.append(a)

                # Mengurutkan nilai di dalam neighbors
                neighbors.sort()

                # Ambil nilai median pada posisi ke-24 (index array Python dimulai dari 0)
                nilai_median = neighbors[24]

                # Posisikan nilai piksel sebelumnya dengan nilai median
                img_out[i, j] = nilai_median

        # Menyimpan dan menampilkan citra hasil median ke GUI
        self.processed_image = img_out
        self.displayImage(self.processed_image, self.hasilLabel)

    def brightness(self):
        if self.image is None:
            return

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        img_out = img.copy()

        for i in range(h):
            for j in range(w):
                pixel = img[i, j] + 50
                if pixel > 255:
                    pixel = 255
                img_out[i, j] = pixel

        self.processed_image = img_out
        self.displayImage(self.processed_image, self.hasilLabel)


    def contrast(self):
        if self.image is None:
            return

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        img_out = img.copy()

        for i in range(h):
            for j in range(w):
                pixel = img[i, j] * 1.5
                if pixel > 255:
                    pixel = 255
                img_out[i, j] = pixel

        self.processed_image = img_out
        self.displayImage(self.processed_image, self.hasilLabel)


    def negative(self):
        if self.image is None:
            return

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        img_out = img.copy()

        for i in range(h):
            for j in range(w):
                img_out[i, j] = 255 - img[i, j]

        self.processed_image = img_out
        self.displayImage(self.processed_image, self.hasilLabel)


    def threshold(self):
        if self.image is None:
            return

        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        img_out = img.copy()

        T = 128

        for i in range(h):
            for j in range(w):
                if img[i, j] < T:
                    img_out[i, j] = 0
                else:
                    img_out[i, j] = 255

        self.processed_image = img_out
        self.displayImage(self.processed_image, self.hasilLabel)

    def displayImage(self, img_array, target_label):
        if img_array is None:
            return

        # Ambil dimensi gambar
        if len(img_array.shape) == 3:  # Citra Berwarna (RGB/BGR)
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            img_display = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            format_qt = QImage.Format_RGB888
        else:  # Citra Grayscale
            h, w = img_array.shape
            bytes_per_line = w
            img_display = img_array
            format_qt = QImage.Format_Grayscale8

        q_img = QImage(img_display.data, w, h, bytes_per_line, format_qt)

        target_label.setPixmap(QPixmap.fromImage(q_img))
        target_label.setScaledContents(True)
        target_label.setAlignment(QtCore.Qt.AlignCenter)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = ShowImage()  # Membuat instance dari class ShowImage
    window.setWindowTitle('PCD - Show Image GUI')

    window.show()
    sys.exit(app.exec_())
