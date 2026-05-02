from imageai.Detection import ObjectDetection
from uuid import uuid4
from datetime import datetime
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import os

def ai_classification(name_img, name_folder="cut_img", output_img="imageout.jpg"):
    # Задаем модель с ImageAi
    os.makedirs(name_folder, exist_ok=True)
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("yolov3.pt")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=name_img,
                                                output_image_path=output_img,
                                                minimum_percentage_probability=30)
    # Считываем изображения
    image = cv2.imread(name_img)
    # Идем по объектам, которые нашла нейросеть
    for number, eachObject in enumerate(detections):
        start_point = (eachObject["box_points"][0], eachObject["box_points"][1])
        end_point = (eachObject["box_points"][2], eachObject["box_points"][3])
        # Если нужный нам класс, мы вырезаем его в заданную папку
        if eachObject["name"] in 'bird  ':
            cropped_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            uid = uuid4().hex[:8]  # короткий уникальный хвост
            ts = datetime.now().strftime('%Y%m%d-%H%M%S-%f')  # метка времени
            out_name = f"{name_img}_{number}_{ts}_{uid}.png"
            cv2.imwrite(f"{name_folder}/{out_name}_{number}.png", cropped_image)

    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]
    # поддержка формата "0 class_name"
    class_names = [line.split(maxsplit=1)[-1] for line in raw]
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    cut_imgs = os.listdir(name_folder)

    results = {}

    for i in range(len(cut_imgs)):
        image = Image.open(f'{name_folder}/{cut_imgs[i]}').convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        results[f'{name_folder}/{cut_imgs[i]}'] = [class_name[2:len(class_name)-1], round(confidence_score*100)]

    return results