import cv2
import numpy as np


def start_point():

    # Задание 1. Прочитать изображение с камеры и перевести его в формат HSV.
    cap = cv2.VideoCapture(0) # Создаем объект cap для видеопотока с веб-камеры
                              # (устройство с индексом 0). Это может быть встроенная
                              # камера или другое устройство, подключенное к компьютеру.
    while True:               # Начинаем бесконечный цикл для чтения и обработки каждого кадра из видеопотока.
        ret, frame = cap.read() #Читаем следующий кадр из видеопотока. ret - флаг успешности чтения, frame - считанный кадр.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Преобразуем цветовое пространство изображения из BGR
        # (цветовое пространство по умолчанию в OpenCV) в HSV (оттенок, насыщенность, значение).
        # Результат сохраняется в переменной hsv. Теперь hsv представляет собой изображение в цветовом пространстве HSV.


        # Задание 2. Применить фильтрацию изображения с помощью команды
        # inRange и оставить только красную часть, вывести получившееся изображение
        # на экран(treshold), выбрать красный объект и потестировать параметры
        # фильтрации, подобрав их нужного уровня.


        lower_red = np.array([160, 100, 100])  # Задаем переменную lower_red и присваиваем ей массив NumPy,
        # представляющий нижнюю границу диапазона красного цвета в формате HSV. В данном случае, это оттенок,
        # насыщенность и значение для красного цвета.
        upper_red = np.array([180, 255, 255])  # Задаем переменную upper_red и присваиваем ей массив NumPy,
        # представляющий верхнюю границу диапазона красного цвета в формате HSV. В данном случае, это оттенок,
        # насыщенность и значение для красного цвета.
        # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение
        # 255 (белый), а остальные пиксели имеют значение 0 (черный).
        mask = cv2.inRange(hsv, lower_red, upper_red)  # Создаем маску (mask), используя функцию cv2.inRange,
        # которая выделяет области изображения в формате HSV, соответствующие заданному диапазону цвета.
        # В данном случае, маска будет содержать значения, соответствующие красному цвету в изображении в формате HSV.
        red_only = cv2.bitwise_and(frame, frame, mask=mask)  # Применяем маску mask к исходному изображению (frame) с
        # использованием функции cv2.bitwise_and. Это позволяет выделить только те области изображения,
        # которые соответствуют красному цвету в формате HSV, остальные области становятся черными.
        # Результат сохраняется в переменной


        # Задание 3. Провести морфологические преобразования (открытие и
        # закрытие) фильтрованного изображения, вывести результаты на экран,
        # посмотреть смысл подобного применения операций erode и dilate

        kernel = np.ones((5, 5), np.uint8)  # Создаем ядро (kernel) для операций морфологии.
        # В данном случае, создается матрица размером 5x5, заполненная единицами, и используется тип данных
        # np.uint8 (8-битное беззнаковое целое число).

        # erosion = cv2.erode(mask, kernel, iterations=1)  # Эрозия
        # dilation = cv2.dilate(mask, kernel, iterations=1)  # Диляция

        # Метод cv2.morphologyEx в OpenCV предназначен для выполнения различных морфологических операций на
        # изображениях. Он предоставляет общий интерфейс для применения операций, таких как эрозия, диляция,
        # открытие, закрытие и других, с использованием ядра (kernel).
        image_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Открытие (Erosion followed by Dilation)
        image_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Закрытие (Dilation followed by Erosion)
        #mask: Исходное изображение или маска, к которой будет применена операция.
        #cv2.MORPH_OPEN: Тип морфологической операции. В данном случае, это операция открытия, которая включает в себя сначала эрозию, а затем диляцию.
        #kernel: Ядро (kernel), которое определяет форму или размер операции.

        # Задание 4. Найти моменты на полученном изображении 1 первого
        # порядка, найти площадь объекта.
    # m00 — это количество всех точек, составляющих объект
        moments = cv2.moments(mask)  # Вычисление моментов изображения для маски
        area = moments['m00']  # Извлечение общей площади объекта на изображении
        #'m00' в контексте моментов изображения представляет собой нулевой центральный момент массы объекта на изображении.
        # Этот момент используется для вычисления различных характеристик объекта, таких как его площадь,
        # координаты центра масс, момент инерции и других.
        # В формулах моментов:
        #'m00' - это интегральная мера интенсивности объекта на изображении. В частности, это сумма значений пикселей внутри объекта.
        # Центральные моменты используются для вычисления центра масс объекта и других его характеристик. В данном случае,
        # 'm00' используется для вычисления площади объекта на изображении.

        # Задание 5 На основе анализа площади объекта найти его центр и
        # построить черный прямоугольник вокруг объекта. Сделать так, чтобы на видео
        # выводился полученный черный прямоугольник, причем на новом кадре
        # новый.

        # 'm10' Этот момент представляет собой сумму произведений интенсивности пикселей на их координаты х
        # Используется для вычисления координаты центра масс объекта по оси

        # 'm01'Этот момент представляет собой сумму произведений интенсивности пикселей на их координаты у
        # Используется для вычисления координаты центра масс объекта по оси

        # if area > 0: # Проверка, что площадь объекта больше 0, что указывает на наличие объекта на изображении
        #     width = height = int(np.sqrt(area))  # Вычисление ширины и высоты объекта на основе площади.
        #                                          # В данном случае, ширина и высота прямоугольника приближаются к квадрату.
        #     # Вычисление координат центра масс объекта по осям X и Y соответственно
        #     c_x = int(moments["m10"] / moments["m00"])  # Вычисление координаты центра масс по X
        #     c_y = int(moments["m01"] / moments["m00"])  # Вычисление координаты центра масс по Y
        #
        #
        #     # Рисование прямоугольника вокруг объекта с использованием координат центра масс и вычисленных ширины и высоты.
        #
        #     cv2.rectangle(
        #         frame,
        #         (c_x - (width // 16), c_y - (height // 16)),  # Верхний левый угол
        #         (c_x + (width // 16), c_y + (height // 16)),  # Нижний правый угол
        #         (0, 0, 0),  # Цвет прямоугольника (черный в формате BGR)
        #         2  # Толщина линии прямоугольника
        #     )
        if area > 0:

            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])

            size = int(np.sqrt(area) / 2)

            points = [
                (c_x, c_y - size),
                (c_x + size * np.sin(72 * np.pi / 90), c_y - size * np.cos(72 * np.pi / 90)),
                (c_x + size * np.sin(144 * np.pi / 90), c_y - size * np.cos(144 * np.pi / 90)),
                (c_x - size * np.sin(144 * np.pi / 90), c_y - size * np.cos(144 * np.pi / 90)),
                (c_x - size * np.sin(72 * np.pi / 90), c_y - size * np.cos(72 * np.pi / 90))
            ]


            cv2.polylines(
                frame,
                [np.array(points, dtype=np.int32)],
                True,  # Замкнутая фигура
                (0, 0, 0),  # Цвет пентаграммы (черный в формате BGR)
                2  # Толщина линий пентаграммы
            )





        # Отображение результирующих кадров
        cv2.imshow('frame', frame)            # Исходное изображение с нарисованным прямоугольником
        cv2.imshow('hsv', hsv)                # Задание 1 HCV
        cv2.imshow('Red Only', red_only)      # Задание 2 Изображение с выделенной красной областью
        cv2.imshow("Opening", image_opening)  # Результат операции открытия
        cv2.imshow("Closing", image_closing)  # Результат операции закрытия

        # Дополнительные отображения (закомментированные строки)
        # cv2.imshow('Erosion', erosion)  # Изображение после операции эрозии
        # cv2.imshow('Dilation', dilation)  # Изображение после операции диляции

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(area)

start_point()




