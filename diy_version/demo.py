"""

this file will give out a simple background subtraction case achieved by GMM
"""

from diy_version.bgst import *
import cv2

if __name__ == '__main__':
    test_frame = 247  # the test frame for testing the effect
    test_dir = '/home/jrq/Downloads/data/hand_segmented_00247.BMP'
    cap = cv2.VideoCapture('/home/jrq/Downloads/data/WavingTrees/b%05d.bmp')
    gmm = createBackgroundSubtractorGMM(history=200)
    i = 0
    while (1):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if ret is True:
            fgmask = gmm.apply(frame, i)
            cv2.imshow('mask', fgmask)
            cv2.imshow('original', frame)
        else:
            break
        k = cv2.waitKey(30)
        if i == 247:
            temp_frame = cv2.imread(test_dir)
            cv2.imshow('test', temp_frame)
            cv2.waitKey(0)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
