import cv2 as cv
import numpy as np

video_source = 'data/VID_20210311_101140.mp4'
darknet_config = 'custom-yolov4-tiny-detector.cfg'
darknet_weight = 'custom-yolov4-tiny-detector_last.weights'
# confidence level
conf = 0.5
rescale_factor = 50  # in percentage

# opencv setting + var
WHITE = (255, 255, 255)
img = None
outputs = None

vid_src = cv.VideoCapture(video_source)
vid_src.set(10, 150)

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet(darknet_config, darknet_weight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# press any key to quit
while cv.waitKey(1) == -1:
    grabbed, img0 = vid_src.read()
    img = img0.copy()

    # preprocess the image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

    outputs = np.vstack(outputs)

    # process Images
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w // 2), int(y - h // 2)
            p1 = int(x + w // 2), int(y + h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    # scale output video
    scale_percent = rescale_factor  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # show output image
    cv.imshow('window', img)


vid_src.release
cv.destroyAllWindows()
