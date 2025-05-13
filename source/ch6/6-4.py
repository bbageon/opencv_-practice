import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QSound
import sys
import os

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러옴', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [['child.png', '어린이'], ['elder.png', '노인'], ['disabled.png', '장애인']]
        self.signImgs = []
        self.roadImg = None

    def signFunction(self):
        self.label.setText('교통약자 번호판을 등록합니다.')
        self.signImgs.clear()

        for fname, _ in self.signFiles:
            img = cv.imread(fname)
            if img is not None:
                self.signImgs.append(img)
                cv.imshow(fname, img)
            else:
                self.label.setText(f'{fname} 파일을 찾을 수 없습니다.')

    def roadFunction(self):
        if not self.signImgs:
            self.label.setText('먼저 번호판을 등록하세요.')
            return

        fname, _ = QFileDialog.getOpenFileName(self, '파일 읽기', './')
        if fname:
            self.roadImg = cv.imread(fname)
            if self.roadImg is None:
                self.label.setText('파일을 읽을 수 없습니다.')
            else:
                cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
            return

        sift = cv.SIFT_create()
        KD = []

        for img in self.signImgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            KD.append(sift.detectAndCompute(gray, None))

        grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        road_kp, road_des = sift.detectAndCompute(grayRoad, None)

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        GM = []

        for sign_kp, sign_des in KD:
            knn_match = matcher.knnMatch(sign_des, road_des, 2)
            T = 0.7
            good_match = [m[0] for m in knn_match if len(m) == 2 and m[0].distance / m[1].distance < T]
            GM.append(good_match)

        best = GM.index(max(GM, key=len))

        if len(GM[best]) < 4:
            self.label.setText('표지판이 없습니다.')
        else:
            sign_kp = KD[best][0]
            good_match = GM[best]

            points1 = np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
            points2 = np.float32([road_kp[gm.trainIdx].pt for gm in good_match])

            H, _ = cv.findHomography(points1, points2, cv.RANSAC)

            h1, w1 = self.signImgs[best].shape[:2]
            h2, w2 = self.roadImg.shape[:2]

            box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
            box2 = cv.perspectiveTransform(box1, H)

            self.roadImg = cv.polylines(self.roadImg, [np.int32(box2)], True, (0, 255, 0), 4)

            img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            cv.drawMatches(self.signImgs[best], sign_kp, self.roadImg, road_kp, good_match, img_match,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imshow('Matches and Homography', img_match)

            self.label.setText(self.signFiles[best][1] + ' 보호구역입니다. 30km로 서행하세요.')

            # 소리 재생 (macOS 호환)
            QSound.play("/System/Library/Sounds/Glass.aiff")  # mac 기본 사운드 예시

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TrafficWeak()
    win.show()
    sys.exit(app.exec_())
