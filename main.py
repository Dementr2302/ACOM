import cv2
import numpy as np

file_name = 'img.png'


def task2():
    image1 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('Window 1', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Window 1', image1)
    cv2.waitKey(0)

    image2 = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cv2.namedWindow('Window 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Window 2', image2)
    cv2.waitKey(0)

    image3 = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    cv2.namedWindow('Window 3', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Window 3', image3)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



def task3():
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Проверка, был ли успешно открыт видеофайл
    if not cap.isOpened():
        print("Ошибка: видеофайл не найден или не может быть открыт.")
        return

    while True:
        ret, frame = cap.read()

        # Проверка, является ли кадр действительным
        if ret:
            cv2.imshow('Video', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task4():
    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Проверка, был ли успешно открыт видеофайл
    if not cap.isOpened():
        print("Ошибка: видеофайл не найден или не может быть открыт.")
        return

    output_video_path = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (w, h))

    while True:
        ret, frame = cap.read()

        # Проверка, является ли кадр действительным
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def task5():
    image = cv2.imread(file_name)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imshow('Original Image', image)
    cv2.imshow('HSV Image', hsv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task7():
    video = cv2.VideoCapture(0)
    _, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    video_writer = cv2.VideoWriter(
        "/Users/dmitrijbajramov/Desktop/ИПИС/video1.mp4", fourcc, 25, (w, h)
    )

    while True:
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

def task6():
    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()

        height, width, _ = frame.shape

        cross_image = np.copy(frame)
        cv2.rectangle(cross_image, (width // 2 - 100, height // 2 - 20), (width // 2 + 100, height // 2 + 20),
                      (0, 0, 255), 15)
        cv2.rectangle(cross_image, (width // 2 - 20, height // 2 - 100), (width // 2 + 20, height // 2 + 100),
                      (0, 0, 255), 15)

        cv2.imshow('Red Cross', cross_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task8():
    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Преобразование цвета центрального пикселя в HSV
        hsv_color = cv2.cvtColor(np.uint8([[frame[center_y, center_x]]]), cv2.COLOR_BGR2HSV)[0][0]

        if hsv_color[0] >= 0 and hsv_color[0] < 20:
            # Ближе к красному (по HSV)
            fill_color = (0, 0, 255)  # Красный
        elif hsv_color[0] >= 40 and hsv_color[0] < 80:
            # Ближе к зеленому (по HSV)
            fill_color = (0, 255, 0)  # Зеленый
        else:
            # Ближе к синему (по HSV)
            fill_color = (255, 0, 0)  # Синий

        cv2.rectangle(frame, (width // 2 - 100, height // 2 - 20), (width // 2 + 100, height // 2 + 20),
                      fill_color, 15)
        cv2.rectangle(frame, (width // 2 - 20, height // 2 - 100), (width // 2 + 20, height // 2 + 100),
                      fill_color, 15)

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task9():
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # task2()
    # task3()
    # task4()
    # task5()
    # task6()
    # task7()
    task8()
    # task9()
