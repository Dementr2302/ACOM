import cv2
import numpy as np

image_path = 'img.png'
# В данной работе контур рассматривается как совокупность пикселей, в
# окрестности которых наблюдается скачкообразное изменение функции
# яркости. Точки контура представляют собой границу объекта, отделяющую
# его от фона. В дальнейшем в данной работе также для обозначения контура
# будет использоваться понятие граница, подразумевающее границу яркости.
def process_and_blur_image(image_path):
    try:
        # Читаем изображение
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Переводим изображение в черно-белый формат
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применяем размытие по Гауссу
        blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)  # Меняйте размер матрицы и sigma по вашему желанию

        # Выводим исходное изображение
        cv2.imshow('Task 1 - Original Image', grayscale_image)
        cv2.waitKey(0)

        # Выводим размытое изображение
        cv2.imshow('Task 1 - Blurred Image', blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

# Сглаживание изображения с помощью фильтра Гаусса.
# Вычисление градиента интенсивности с использованием операторов Собеля.
# Подавление немаксимумов для выделения локальных максимумов в градиенте.
# Двойная пороговая фильтрация для определения пикселей границ.
# Связывание границ и удаление шума.
# - cv2.CV: тип выходных данных градиента. В данном случае, используется 64-битное число с плавающей запятой для точности.
# - 1: порядок производной по оси x.
# Суть операции свертки заключается в перемещении фильтра по всем пикселям изображения
# и вычислении взвешенной суммы значений пикселей, попадающих под фильтр. Веса определяются
# значениями фильтра. Результатом операции свертки является новое изображение, в котором каждый
# пиксель представляет собой взвешенную сумму значений пикселей вокруг него.
def calculate_gradients(image_path):
    try:
        # Читаем изображение
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Оператор Собеля - это оператор, используемый для вычисления градиента яркости пикселей. Он основан на свертке
        # изображения с двумя ядрами (одним для вычисления горизонтального градиента и другим для вертикального).
        # Оператор Собеля выявляет вертикальные и горизонтальные границы на изображении.
        # Вычисляем градиенты по x и y направлениям
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # Этап подавления немаксимумов заключается в сравнении значений градиента пикселя с его соседями в направлении градиента.
        # Только локальные максимумы градиента сохраняются, а все остальные пиксели подавляются.
        # Угол градиента играет роль в определении направления проверки для каждого пикселя.
        # Вычисляем длины и углы градиентов
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi  # Перевод в градусы

        # Выводим матрицу длин градиентов
        cv2.imshow('Task 2 - Gradient Magnitude', gradient_magnitude.astype(np.uint8))
        cv2.waitKey(0)

        # Выводим матрицу углов градиентов
        cv2.imshow('Task 2 - Gradient Angle', gradient_angle.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


def non_max_suppression(image_path):
    try:
        # Читаем изображение в оттенках серого
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Вычисляем градиенты по x и y направлениям
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Вычисляем длины и углы градиентов
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi  # Перевод в градусы

        # Выполняем подавление немаксимумов
        suppressed_image = np.zeros_like(gradient_magnitude)
        height, width = gradient_magnitude.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                angle = gradient_angle[i, j]

                # Определяем соседние пиксели в направлении градиента
                if (angle >= -22.5 and angle < 22.5) or (angle >= 157.5 or angle < -157.5):
                    neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
                elif (angle >= 22.5 and angle < 67.5) or (angle >= -157.5 and angle < -112.5):
                    neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
                elif (angle >= 67.5 and angle < 112.5) or (angle >= -112.5 and angle < -67.5):
                    neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

                # Проверяем, является ли текущий пиксель локальным максимумом
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        # Выводим изображение с подавлением немаксимумов
        cv2.imshow('Task 3 - Non-Maximum Suppression', suppressed_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")





def double_thresholding(image_path, low_threshold, high_threshold):
    try:
        # Читаем изображение в оттенках серого
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Вычисляем градиенты по x и y направлениям
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Вычисляем длины градиентов
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Применяем двойную пороговую фильтрацию
        edge_map = np.zeros_like(image)
        strong_edges = (gradient_magnitude >= high_threshold)
        weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold)
        edge_map[strong_edges] = 255
        edge_map[weak_edges] = 127

        # Выводим изображение с примененной двойной пороговой фильтрацией
        cv2.imshow('Task 4 - Edge Map', edge_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


def double_thresholding2(image_path, low_threshold, high_threshold, kernel_size, sigma):
    try:
        # Читаем изображение в оттенках серого
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Применяем размытие Гаусса
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Вычисляем градиенты по x и y направлениям
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Вычисляем длины градиентов
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Применяем двойную пороговую фильтрацию
        edge_map = np.zeros_like(image)
        strong_edges = (gradient_magnitude >= high_threshold)
        weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude < high_threshold)
        edge_map[strong_edges] = 255
        edge_map[weak_edges] = 127

        # Выводим изображение с примененной двойной пороговой фильтрацией
        cv2.imshow('Edge Map', edge_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


process_and_blur_image(image_path)


calculate_gradients(image_path)

non_max_suppression(image_path)

low_threshold = 50  # Нижний порог
high_threshold = 150  # Верхний порог
double_thresholding(image_path, low_threshold, high_threshold)


low_threshold = 50  # 10 10 100 Нижний порог
high_threshold = 100  # 3 11 33 Верхний порог
kernel_size = 5  # 15 6 55 Размер матрицы Гаусса
sigma = 1.0  # Значение среднеквадратичного отклонения

double_thresholding2(image_path, low_threshold, high_threshold, kernel_size, sigma)

# task5
# можно изменять значения low_threshold, high_threshold, kernel_size и sigma
# и прогонять изображение через процесс обработки с разными параметрами
