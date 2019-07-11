
import cv2
import target
import kbd
import eye_detector as ed
import head_tracker as ht
import grid
import numpy as np
import time
import os
from multiprocessing import Process, Pipe, Condition
        

#================================================
#
#------------------------------------------------
if __name__=="__main__":
    #Loading and preparing everything
    head_tracker = ht.HeadTracker("TF")
    eye_detector = ed.EyeDetector()
    processor = Processor(45)
    cam_id = 0
    cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    head_grid = grid.Grid(w, h, 20, 15)
    p1, c1 = Pipe()
    p2, c2 = Pipe()
    keyboard = kbd.Keyboard(c1)
    tgt = target.TargetScreen(1920, 1080, 'target.png')
    tgt_proc = Process(target=tgt.run, args=(c2,))
    tgt_proc.start()
    store_data = False
    coord = ""
    s = 112
    count = 0

    #Streaming loop
    while cam.isOpened():

        #triggering data recording
        if p1.poll():
            msg = p1.recv()
            if msg.startswith("NEXT"):
                p2.send("BREAK")

        if p2.poll():
            msg = p2.recv()
            if msg == "END":
                store_data = False
                #saving everything. It takes a while...
                processor.save_data(coord, "cadu6", "train")
            else:
                store_data = True
                coord = msg

        ret, frame = cam.read()
        if ret:
            frame = processor.gamma_correction(frame, 0.35)
            frame_dnn, bboxes = head_tracker.detect_head(frame)
            if bboxes:
                b = bboxes[0]
                if len(bboxes) > 1:
                    b = bboxes[1]
                head = processor.crop_img(b, frame)
                head_dnn = processor.crop_img(b, frame_dnn)
                head_img = processor.resize_img(head, s)
                f,l,r = eye_detector.detect_features(head_dnn)
                if f and l and r:
                    le_crop = processor.crop_img(l, head)
                    re_crop = processor.crop_img(r, head)
                    le_img  = processor.resize_img(le_crop, s)
                    re_img  = processor.resize_img(re_crop, s)
                    res_grid = head_grid.get_intersection(b)
                    if store_data:
                        #stat = processor.bundle_data(f, le_img, re_img, res_grid)
                        stat = processor.bundle_data(head_img, le_img, re_img, res_grid)
                        if stat == "full":
                            p2.send("BREAK")
                            time.sleep(0.1)

                    # cv2.imwrite('test/left_eye' + str(count) + '.jpg', le_img)
                    # cv2.imwrite('test/right_eye' + str(count) + '.jpg', re_img)
                    # count += 1
                #     cv2.imshow('left_eye', le_img)
                #     cv2.imshow('right_eye', re_img)
                # cv2.imshow('head', head_img)
            
            cv2.imshow("window", frame_dnn)
            k = cv2.waitKey(2)
            if k == 27:
                p2.send("QUIT")
                break

    tgt_proc.join()
    cv2.destroyAllWindows()
