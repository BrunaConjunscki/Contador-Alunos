import cv2

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Inicia a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Detecção de objetos no frame
    (h, w) = frame.shape[:2]
    redimensionado = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(redimensionado, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    deteccoes = net.forward()
    
    num_pessoas = 0
    for i in range(deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]
        idx = int(deteccoes[0, 0, i, 1])

        if CLASSES[idx] == "person" and confianca > 0.4:
            num_pessoas += 1

            box = deteccoes[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.putText(frame, f"Pessoas: {num_pessoas}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibe o resultado
    cv2.imshow("Contador de Pessoas (Webcam)", frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) == 27:
        break

# Libera a webcam
cap.release()
cv2.destroyAllWindows()
