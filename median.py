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
        self.actionmean1_2.triggered.connect(self.mean1)
        self.actionmean2_2.triggered.connect(self.mean2)
        self.actiongaus.triggered.connect(self.gaus)

    @pyqtSlot()
    def loadClicked(self):
        options = QFileDialog.Options()
        flname, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp)",
                                                options=options)

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

        # Mengubah gambar asli dari berwarna (BGR) menjadi grayscale (hitam putih)
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        img_out = img_gray.copy()

        # Mengambil informasi tinggi (h) dan lebar (w) dari gambar
        h, w = img_gray.shape[:2]

        # Mengabaikan 3 piksel terluar di semua sisi gambar (atas, bawah, kiri, kanan)
        for i in range(3, h - 3):
            for j in range(3, w - 3):

                # List kosong untuk menampung nilai intensitas 49 piksel di sekitar titik pusat
                neighbors = []

                # 3. MEMBACA PIKSEL DALAM KERNEL 7x7

                for k in range(-3, 4):
                    for l in range(-3, 4):
                        # Mengambil nilai intensitas piksel tetangga satu per satu
                        a = img_gray[i + k, j + l]

                        neighbors.append(a)

                # Mengurutkan ke-49 nilai piksel dari yang paling gelap ke yang paling terang
                neighbors.sort()

                # Mengambil nilai tengah (median) dari 49 data yang sudah diurutkan.
                nilai_median = neighbors[24]

                img_out[i, j] = nilai_median

        self.processed_image = img_out

        self.displayImage(self.processed_image, self.hasilLabel)

    def mean(self, X, F):

        tinggi_citra, lebar_citra = X.shape[:2]
        tinggi_kernel, lebar_kernel = F.shape[:2]

        # Tentukan titik tengah (anchor) kernel
        # Digunakan untuk menyelaraskan filter dengan pixel yang diproses
        H_anchor = (tinggi_kernel - 1) // 2
        W_anchor = (lebar_kernel - 1) // 2

        # array kosong untuk menampung hasil (output)
        out = np.zeros_like(X, dtype=np.float32)

        # Tentukan batas area yang bisa dikonvolusi

        batas_atas = H_anchor
        batas_bawah = tinggi_citra - (tinggi_kernel - 1 - H_anchor)
        batas_kiri = W_anchor
        batas_kanan = lebar_citra - (lebar_kernel - 1 - W_anchor)

        for i in range(batas_atas, batas_bawah):
            for j in range(batas_kiri, batas_kanan):

                sum_val = 0.0

                # 6. Loop Kernel: Mengalikan kernel dengan area sekitar pixel (i, j)
                for r in range(tinggi_kernel):
                    for c in range(lebar_kernel):
                        # Cari koordinat pixel tetangga yang bersesuaian dengan posisi di kernel
                        pos_x = i + (r - H_anchor)
                        pos_y = j + (c - W_anchor)

                        a = X[pos_x, pos_y]  # Nilai intensitas pixel gambar
                        w = F[r, c]          # Nilai bobot filter/kernel
                        sum_val += (w * a)   # Akumulasi hasil perkalian

                # Simpan hasil akhir perhitungan kernel ke posisi pixel (i, j)
                out[i, j] = sum_val

        # Clip nilai agar tetap di rentang 0-255 dan ubah ke 8-bit integer

        return np.clip(out, 0, 255).astype(np.uint8)

    def mean1(self):

        kernel_3x3 = np.array([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ], dtype=np.float32)

        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.processed_image = self.mean(img_gray, kernel_3x3)
        self.displayImage(self.processed_image, self.hasilLabel)

    def mean2(self):

        kernel_2x2 = np.array([
            [1/4, 1/4],
            [1/4, 1/4]
        ], dtype=np.float32)

        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.processed_image = self.mean(img_gray, kernel_2x2)
        self.displayImage(self.processed_image, self.hasilLabel)

    def konvolusi(self, X, F):
        tinggi_citra, lebar_citra = X.shape[:2]
        tinggi_kernel, lebar_kernel = F.shape[:2]

        H = tinggi_kernel // 2
        W = lebar_kernel // 2

        out = np.zeros_like(X, dtype=np.float32)

        for i in range(H, tinggi_citra - H):
            for j in range(W, lebar_citra - W):
                sum_val = 0
                for k in range(-H, H + 1):
                    for l in range(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum_val += (w * a)

                out[i, j] = sum_val

        return np.clip(out, 0, 255).astype(np.uint8)

    def gaus(self):
        # Menentukan ukuran matriks filter (5x5) dan tingkat kelembutan (sigma)
        kernel_size = 5
        sigma = 1.4

        # Membuat range koordinat dari -2 sampai 2 (untuk kernel size 5)
        ax = np.linspace(-(kernel_size - 1) / 2.,
                         (kernel_size - 1) / 2., kernel_size)

        # Membuat grid koordinat x dan y berdasarkan range
        x, y = np.meshgrid(ax, ax)

        # Menghitung nilai distribusi Gaussian untuk setiap titik di kernel
        kernel = (1 / (2 * np.pi * sigma**2)) * \
            np.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalisasi kernel agar total nilainya = 1 (menjaga kecerahan gambar tetap konstan)
        kernel = kernel / np.sum(kernel)
        kernel = kernel.astype(np.float32)

        # Mengubah gambar BGR (standard OpenCV) ke Grayscale
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Melakukan proses konvolusi manual antara gambar dan kernel Gaussian
        self.processed_image = self.konvolusi(img_gray, kernel)

        # Menampilkan hasil pemrosesan ke UI/Label
        self.displayImage(self.processed_image, self.hasilLabel)

    def displayImage(self, img_array, target_label):
        if img_array is None:
            return

        if len(img_array.shape) == 3:
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
