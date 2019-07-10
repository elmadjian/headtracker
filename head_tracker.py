import cv2


class HeadTracker():

    def __init__(self, dnn):
        self.conf_threshold = 0.7
        self.net = None
        if dnn == "CAFFE":
            model_file  = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "deploy.prototxt"
            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        else:
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    def detect_head(self, frame):
        frame_dnn = frame.copy()
        frame_height = frame_dnn.shape[0]
        frame_width  = frame_dnn.shape[1]
        blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (250,250))
        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            offset = 20
            if confidence > self.conf_threshold:
                x1 = int(detections[0,0,i,3] * frame_width)  - offset
                y1 = int(detections[0,0,i,4] * frame_height) - offset
                x2 = int(detections[0,0,i,5] * frame_width)  + offset
                y2 = int(detections[0,0,i,6] * frame_height) + offset
                bboxes.append([x1,y1,x2,y2])
                cv2.rectangle(frame_dnn, (x1,y1), (x2,y2), (0,0,255), 3)
        return frame_dnn, bboxes

