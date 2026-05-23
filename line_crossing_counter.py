import cv2
import argparse
import sys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Пути по умолчанию ---
DEFAULT_VIDEO = r"C:\Users\Nikita\Desktop\MAGA\Comp_vision\HW-FaceDetection\shutterstock_13018979.mov"
DEFAULT_MODEL = r'C:\Users\Nikita\Desktop\MAGA\Comp_vision\HW-FaceDetection\yolov8n-face.pt'

# Глобальные переменные для интерактивного рисования линии мышью
line_points = []
drawing = False
line_set = False


def draw_line_callback(event, x, y, flags, param):
    """
    Callback мыши: два клика задают начало и конец линии подсчёта.
    После второго клика линия зафиксирована — нажмите пробел для старта.
    """
    global line_points, drawing, line_set
    if line_set:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))
        if len(line_points) == 2:
            line_set = True
            print(f"Линия установлена: {line_points[0]} -> {line_points[1]}")
            print("Нажмите ПРОБЕЛ для продолжения...")
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing and len(line_points) == 1:
        # Предварительный просмотр линии пока тянем мышь
        temp_frame = param.copy()
        cv2.line(temp_frame, line_points[0], (x, y), (0, 0, 255), 2)
        cv2.imshow('Line Crossing Counter', temp_frame)


def is_line_crossed(pt1, pt2, line_start, line_end):
    """
    Проверяет пересекает ли отрезок pt1→pt2 линию line_start→line_end.
    Использует тест ориентации (CCW — Counter-ClockWise):
    два отрезка пересекаются если их концы лежат по разные стороны друг от друга.
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(pt1, line_start, line_end) != ccw(pt2, line_start, line_end) and \
           ccw(pt1, pt2, line_start) != ccw(pt1, pt2, line_end)


def main():
    parser = argparse.ArgumentParser(description="Подсчет людей, пересекающих линию")
    parser.add_argument('--video', type=str, default=DEFAULT_VIDEO, help="Путь к видеофайлу")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="Путь к модели YOLO")
    parser.add_argument('--output', type=str, default='output_counted.mp4')
    parser.add_argument('--conf', type=float, default=0.15, help="Порог уверенности детекции")
    parser.add_argument('--line', type=int, nargs=4,
                        help="Координаты линии x1 y1 x2 y2 (если не указать — рисуем мышью)")
    args = parser.parse_args()

    # Загружаем YOLOv8 модель специально обученную на детекции лиц
    model = YOLO(args.model)

    # DeepSort — трекер: связывает детекции одного человека между кадрами,
    # присваивает уникальный ID и не теряет его при кратковременном исчезновении
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {args.video}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        return

    # Линию можно передать через аргументы (--line x1 y1 x2 y2) или нарисовать мышью
    if args.line is not None:
        line_pt1 = (args.line[0], args.line[1])
        line_pt2 = (args.line[2], args.line[3])
        print(f"Используется линия из аргументов: {line_pt1} -> {line_pt2}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # Интерактивный режим: показываем первый кадр, ждём двух кликов
        global line_points, line_set
        line_points = []
        line_set = False

        cv2.namedWindow('Line Crossing Counter')
        cv2.setMouseCallback('Line Crossing Counter', draw_line_callback, first_frame.copy())

        print("На первом кадре нарисуйте линию: кликните дважды (начало и конец).")
        print("После установки линии нажмите ПРОБЕЛ для старта обработки видео.")

        while True:
            display_frame = first_frame.copy()
            if len(line_points) == 1:
                cv2.circle(display_frame, line_points[0], 5, (0, 255, 0), -1)
            elif len(line_points) == 2:
                cv2.circle(display_frame, line_points[0], 5, (0, 255, 0), -1)
                cv2.circle(display_frame, line_points[1], 5, (0, 255, 0), -1)
                cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)

            cv2.imshow('Line Crossing Counter', display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(' ') and line_set:
                break
            elif key == ord('q') or key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        line_pt1 = line_points[0]
        line_pt2 = line_points[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # перематываем на начало

    # Настраиваем запись результата
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    total_count = 0       # общее количество пересечений линии
    tracked_paths = {}    # словарь: track_id → предыдущая позиция центра

    print("Обработка видео. Нажмите 'q' или ESC для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция лиц через YOLOv8
        results = model(frame, conf=args.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            if cls_id == 0:  # класс 0 = лицо
                w, h = x2 - x1, y2 - y1
                # DeepSort ожидает формат [x, y, w, h]
                detections.append(([x1, y1, w, h], conf, 'face'))

        # Обновляем трекер — получаем треки с уникальными ID
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue  # трек ещё не стабилизирован (нужно n_init=3 кадров)

            track_id = track.track_id
            ltrb = track.to_ltrb()  # bounding box в формате left-top-right-bottom
            center_x = int((ltrb[0] + ltrb[2]) / 2)
            center_y = int((ltrb[1] + ltrb[3]) / 2)
            current_pos = (center_x, center_y)

            # Рисуем бокс и ID трека
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (int(ltrb[0]), int(ltrb[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Проверяем: пересёк ли трек линию между предыдущим и текущим кадром
            if track_id in tracked_paths:
                prev_pos = tracked_paths[track_id]
                if is_line_crossed(prev_pos, current_pos, line_pt1, line_pt2):
                    total_count += 1
                    print(f"ID {track_id} пересек линию! Всего: {total_count}")
            tracked_paths[track_id] = current_pos

        # Рисуем линию подсчёта и счётчик на кадре
        cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 3)
        cv2.putText(frame, f'Total Crossed: {total_count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Line Crossing Counter', frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty('Line Crossing Counter', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработка завершена. Результат сохранён в {args.output}")
    print(f"Итоговое количество пересечений: {total_count}")


if __name__ == '__main__':
    main()
