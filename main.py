import cv2
import numpy as np


def load_yolo():
    # Ładowanie modelu YOLO
    net = cv2.dnn.readNetFromDarknet("yolo/yolov4.cfg", "yolo/yolov4.weights")
    return net

def load_classes():
    # Pobieranie nazw klas
    classes = []
    with open("yolo/darknet/cfg/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def detect_objects(frame, net, output_layers, classes):
    height, width, channels = frame.shape
    # Detekcja obiektów za pomocą YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Przetwarzanie wyników detekcji
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.50 and class_id in [0, 2, 56, 62, 63, 64, 66, 67]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Współrzędne lewego górnego rogu prostokąta
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_predictions(frame, boxes, confidences, class_ids, classes):
    # Zastosowanie non-maximum suppression dla zidentyfikowanych obiektów
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Rysowanie ramek i oznaczenie obiektów
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Kolor ramki (zielony)

            # Rysowanie ramki
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Dodawanie etykiety i prawdopodobieństwa
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)
    return frame

def main():
    net = load_yolo()
    classes = load_classes()

    # Ustalenie warstw wyjściowych
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Ustawienia wideo
    video = cv2.VideoCapture("auto.mp4")  # Można podać nazwę pliku lub indeks kamery

    frame_count = 0
    start_time = cv2.getTickCount()
    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        boxes, confidences, class_ids = detect_objects(frame, net, output_layers, classes)
        frame = draw_predictions(frame, boxes, confidences, class_ids, classes)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if frame_count % 10 == 0:
            end_time = cv2.getTickCount()
            elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
            fps = 10 / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = cv2.getTickCount()

    # Zwolnienie zasobów i zamknięcie okna
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()