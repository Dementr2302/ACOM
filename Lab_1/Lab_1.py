import cv2
import numpy as np


def start_point():

    # Задание 1. Прочитать изображение с камеры и перевести его в формат HSV.
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)




        # Задание 2. Применить фильтрацию изображения с помощью команды
        # inRange и оставить только красную часть, вывести получившееся изображение
        # на экран(treshold), выбрать красный объект и потестировать параметры
        # фильтрации, подобрав их нужного уровня.


        lower_red = np.array([160, 100, 100])

        upper_red = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        red_only = cv2.bitwise_and(frame, frame, mask=mask)



        # Задание 3. Провести морфологические преобразования (открытие и
        # закрытие) фильтрованного изображения, вывести результаты на экран,
        # посмотреть смысл подобного применения операций erode и dilate

        kernel = np.ones((5, 5), np.uint8)
        image_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        image_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Задание 4. Найти моменты на полученном изображении 1 первого
        # порядка, найти площадь объекта.

        moments = cv2.moments(mask)
        area = moments['m00']

        if area > 0:
            width = height = int(np.sqrt(area))
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])

            cv2.rectangle(
                frame,
                (c_x - (width // 16), c_y - (height // 16)),
                (c_x + (width // 16), c_y + (height // 16)),
                (0, 0, 0),
                2
            )

        # Отображение результирующих кадров
        cv2.imshow('frame', frame)            # Исходное изображение с нарисованным прямоугольником
        cv2.imshow('Phsv', hsv)                # Задание 1 HCV
        cv2.imshow('Red Only', red_only)      # Задание 2 Изображение с выделенной красной областью
        cv2.imshow("Opening", image_opening)  # Результат операции открытия
        cv2.imshow("Closing", image_closing)  # Результат операции закрытия

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


start_point()




