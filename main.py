import head_tracker as ht
import face_detector as fd
import image_processor as ip
import grid
import cv2
import i3
import time
import numpy as np

def horiz_pos(val, width, min_x, max_x):
    k1 = width/(max_x-min_x)
    k2 = k1*min_x
    if val < min_x: val = min_x
    if val > max_x: val = max_x
    px = val * k1 - k2
    return int(px)

def vert_pos(val, height, min_y, max_y):
    k1 = height/(max_y-min_y)
    k2 = k1*min_y
    if val < min_y: val = min_y
    if val > max_y: val = max_y
    py = val * k1 - k2
    return int(py)


def main(cam_id):
    head_tracker = ht.HeadTracker('TF')
    face_detector = fd.FaceDetector()
    img_processor = ip.ImageProcessor()
    cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 15)
    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    head_grid = grid.Grid(w, h, 20, 15)
    buffer, mx, my = [], 0, 0
    drop = 0

    while cam.isOpened():
        ret,frame = cam.read()
        if drop % 2 == 0:
            drop = (drop + 1) % 2
            continue
        if ret:
            frame = img_processor.gamma_correction(frame, 0.4)
            frame_dnn, bboxes = head_tracker.detect_head(frame)
            if bboxes:
                b = bboxes[-1]
                head = img_processor.crop_img(b, frame)
                f,l,r = face_detector.detect_features(head)
                if f and l and r:
                    print('h:', f[0], 'v:', f[1])
                    #horizontal: center:0.5 - extremes: 0.75,0.25
                    #vertical: center: 0.29 - extremes: 0.22,0.35
                    px = horiz_pos(f[0], 1920, 0.35, 0.65)
                    py = vert_pos(f[1], 1080, 0.25, 0.35)
                    buffer.append(np.array([px,py]))
                    if len(buffer) == 4:
                        mx,my = np.mean(buffer, axis=0)
                        buffer.pop(0)
                    bg = np.ones((1024, 1920, 3), np.uint8) * 255
                    #cv2.circle(bg, (px, py), 20, (0,0,255), -1)
                    cv2.circle(bg, (int(mx), int(my)), 20, (0,255,0), -1)
                    cv2.namedWindow('Test', flags=cv2.WINDOW_GUI_NORMAL)
                    cv2.imshow('Test', bg)

        k = cv2.waitKey(1)
        if k == 27:
            break

def cycle():
    # get currently focused windows
    current = i3.filter(nodes=[], focused=True)
    # get unfocused windows
    other = i3.filter(nodes=[], focused=False)
    # focus each previously unfocused window for 0.5 seconds
    for window in other:
        i3.focus(con_id=window['id'])
        print(window['name'])
        print(window['rect'])
        print('----------')
        time.sleep(0.5)
    # focus the original windows
    # for window in current:
    #     i3.focus(con_id=window['id'])




if __name__=="__main__":
    main(0)
    #kicycle()