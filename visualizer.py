"""
Визуализация работы генетического алгоритма для задачи коммивояжёра.

Читает JSON-файл, созданный программой на C#, и отображает анимацию улучшения маршрута,
а также график сходимости (длина лучшего маршрута от поколения).
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_data(json_file: str) -> list:
    """
    Загружает данные из JSON-файла.

    Параметры:
        json_file (str): Путь к JSON-файлу.

    Возвращает:
        list: Десериализованные данные (список словарей).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_tsp(json_file: str) -> None:
    """
    Основная функция визуализации: создаёт карту маршрута и график сходимости,
    запускает анимацию по поколениям.

    Параметры:
        json_file (str): Путь к JSON-файлу с историей поколений.

    Ожидаемая структура JSON:
        - Первый элемент: метаданные (type='metadata'),
          содержит 'coordinates' (список {'x':..., 'y':...})
          и опционально 'saveStep'.
        - Последующие элементы: type='generation',
          содержат 'generation' (номер), 'bestLength' (длина маршрута),
          'bestRoute' (список городов, замкнутый).
    """
    data = load_data(json_file)
    meta = data[0]
    coords = meta['coordinates']
    xs = [p['x'] for p in coords]
    ys = [p['y'] for p in coords]
    save_step = meta.get('saveStep', 1)

    generations = [item for item in data if item['type'] == 'generation']

    fig, (ax_map, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Левый график: карта городов и маршрут -----
    ax_map.scatter(xs, ys, c='red', s=50, zorder=5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_map.annotate(str(i), (x, y), fontsize=8,
                        ha='center', va='bottom')
    ax_map.set_title("Лучший маршрут")
    ax_map.grid(True, linestyle=':', alpha=0.7)

    # Линия маршрута (будет обновляться в анимации)
    line, = ax_map.plot([], [], 'b-', lw=1.5,
                        marker='o', markersize=3)

    # Текстовое поле с информацией о текущем поколении
    info_text = ax_map.text(0.02, 0.98, '',
                            transform=ax_map.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round',
                                      facecolor='wheat', alpha=0.8))

    # ----- Правый график: сходимость -----
    gen_numbers = [g['generation'] for g in generations]
    best_lengths = [g['bestLength'] for g in generations]
    ax_plot.plot(gen_numbers, best_lengths, 'g-', lw=1)
    ax_plot.set_title("Сходимость")
    ax_plot.set_xlabel("Поколение")
    ax_plot.set_ylabel("Длина маршрута")
    ax_plot.grid(True, linestyle=':', alpha=0.7)

    # Вертикальная линия и точка, отмечающие текущее поколение
    current_gen_line = ax_plot.axvline(x=0, color='r',
                                       linestyle='--', alpha=0.7)
    current_point, = ax_plot.plot([0], [best_lengths[0]], 'ro',
                                  markersize=4)

    # Если данные были сохранены с шагом >1, добавим пояснение
    if save_step > 1:
        ax_plot.text(0.02, 0.98,
                     f"Сохранено каждое {save_step}-е поколение",
                     transform=ax_plot.transAxes, fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='lightyellow'))

    def update(frame: int):
        """
        Функция обновления для анимации.

        Параметры:
            frame (int): Номер текущего кадра (индекс в списке generations).

        Возвращает:
            tuple: Обновляемые объекты графика (линия, текст, вертикальная линия, точка).
        """
        gen = generations[frame]
        route = gen['bestRoute']

        x_route = [xs[city] for city in route]
        y_route = [ys[city] for city in route]

        # Обновляем линию маршрута
        line.set_data(x_route, y_route)

        # Обновляем информационный текст
        info_text.set_text(f"Поколение {gen['generation']}\n"
                           f"Длина = {gen['bestLength']:.2f}")

        # Перемещаем вертикальную линию и точку на графике сходимости
        current_gen_line.set_xdata([gen['generation'], gen['generation']])
        current_point.set_data([gen['generation']], [gen['bestLength']])

        return line, info_text, current_gen_line, current_point

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(generations),
        interval=100,  # 100 мс между кадрами
        repeat=False,
        blit=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Получаем имя JSON-файла из аргументов командной строки или запрашиваем вручную
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = input("Введите путь к JSON файлу: ")

    visualize_tsp(json_file_path)