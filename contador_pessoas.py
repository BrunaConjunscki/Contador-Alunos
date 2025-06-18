import cv2

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

imagem = cv2.imread("sala.jpg")

(h, w) = imagem.shape[:2]
blob = cv2.dnn.blobFromImage(imagem, 0.007843, (300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

count_pessoas = 0

# Percorre todas as detecções
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    idx = int(detections[0, 0, i, 1])

    # Apenas classe "person" com confiança suficiente
    if CLASSES[idx] == "person" and confidence > 0.4:
        count_pessoas += 1

        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(imagem, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Contador de Pessoas", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()