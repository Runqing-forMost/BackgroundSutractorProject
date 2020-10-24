import cv2

"""
background subtract for data WavingTrees
"""
if __name__ == '__main__':
    test_frame = 247  # the test frame for testing the effect
    test_dir = '/home/jrq/Downloads/data/hand_segmented_00247.BMP'
    #  cv2.waitKey(0)
    cap = cv2.VideoCapture('/home/jrq/Downloads/data/WavingTrees/b%05d.bmp')
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200)  # history : number of training frames
    i = 0
    while (1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        if ret is True:
            cv2.imshow('frame', fgmask)
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
