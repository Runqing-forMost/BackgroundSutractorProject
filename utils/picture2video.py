import cv2
import os


"""
generating video from image sequence
"""
if __name__ == '__main__':

    img_root = '/home/jrq/Downloads/data/WavingTrees/'
    fps = 5

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('/home/jrq/Downloads/TestVideo.avi', fourcc, fps, (160, 120), True)

    for idx in sorted(os.listdir(img_root)):
        img = os.path.join(img_root, idx)
        frame = cv2.imread(img)
        videoWriter.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    videoWriter.release()
    cv2.destroyAllWindows()
