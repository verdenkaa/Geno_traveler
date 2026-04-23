using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Diagnostics;

namespace GeneticAlgorithm
{
    /// <summary>
    /// Реализует генетический алгоритм для решения задачи коммивояжёра.
    /// Маршрут представляется замкнутым списком городов, где первый и последний элемент равны.
    /// Алгоритм использует операторы: случайная/жадная инициализация, порядковый кроссовер (OX),
    /// мутация обменом, отбор рабочей группы с заменой худшего.
    /// При необходимости сохраняется история лучших маршрутов в JSON с заданным шагом.
    /// </summary>
    class GeneticAlgorithm
    {
        // Данные задачи
        private readonly int numCities;
        private readonly (double X, double Y)[] coords;
        private readonly double[,] dist;
        private readonly int startCity;
        private readonly Random rand = Random.Shared;

        // Параметры генетического алгоритма
        private readonly int populationSize;
        private readonly int maxGenerations;
        private readonly int wrGroupSize;
        private readonly double mutationChance;
        private readonly int greedyNeighbors;
        private readonly int greedyProbability;

        // Популяция и эволюция
        private List<List<int>> population = new();   // список маршрутов (каждый маршрут – список номеров городов)
        private double[] fitness;                     // длина каждого маршрута (приспособленность)
        private List<int> bestRoute;                  // лучший найденный маршрут
        private double bestLength;                    // длина лучшего маршрута

        // Сохранение истории в JSON
        private readonly string outputJsonPath;
        private readonly int saveStep;
        private readonly List<object> generationRecords = new(); // список записей для JSON


        /// <summary>
        /// Инициализирует генетический алгоритм: вычисляет матрицу расстояний, создаёт начальную популяцию.
        /// </summary>
        /// <param name="numCities">Количество городов.</param>
        /// <param name="coords">Массив координат городов.</param>
        /// <param name="startCity">Индекс стартового города.</param>
        /// <param name="popSize">Размер популяции.</param>
        /// <param name="maxGen">Максимальное число поколений.</param>
        /// <param name="wrGroupSize">Размер рабочей группы.</param>
        /// <param name="mutChance">Вероятность мутации (0..1).</param>
        /// <param name="greedyNeighbors">Количество ближайших соседей для жадного построения.</param>
        /// <param name="greedyProb">Вероятность использования жадного метода (0..100).</param>
        /// <param name="outputJsonPath">Путь для сохранения JSON.</param>
        /// <param name="saveStep">Сохранять каждое N-е поколение (1 – все).</param>
        public GeneticAlgorithm(
            int numCities, (double X, double Y)[] coords, int startCity,
            int popSize, int maxGen, int wrGroupSize, double mutChance,
            int greedyNeighbors, int greedyProb,
            string outputJsonPath = null,
            int saveStep = 1)
        {
            this.numCities = numCities;
            this.coords = coords;
            this.startCity = startCity;
            this.populationSize = popSize;
            this.maxGenerations = maxGen;
            this.wrGroupSize = wrGroupSize;
            this.mutationChance = mutChance;
            this.greedyNeighbors = greedyNeighbors;
            this.greedyProbability = greedyProb;
            this.outputJsonPath = outputJsonPath;
            this.saveStep = saveStep <= 0 ? 1 : saveStep;

            dist = ComputeDistances();

            bestLength = double.MaxValue;
            bestRoute = new List<int>();

            CreateInitialPopulation();

            UpdateBestRoute();
        }

        /// <summary>
        /// Вычисляет евклидовы расстояния между всеми парами городов.
        /// </summary>
        /// <returns>Квадратная матрица расстояний [numCities, numCities].</returns>
        private double[,] ComputeDistances()
        {
            var d = new double[numCities, numCities];
            for (int i = 0; i < numCities; i++)
                for (int j = i + 1; j < numCities; j++)
                {
                    double dx = coords[i].X - coords[j].X;
                    double dy = coords[i].Y - coords[j].Y;
                    double distVal = Math.Sqrt(dx * dx + dy * dy);
                    d[i, j] = d[j, i] = distVal;
                }
            return d;
        }

        /// <summary>
        /// Возвращает полную длину замкнутого маршрута.
        /// </summary>
        /// <param name="route">Маршрут в виде списка городов.</param>
        /// <returns>Суммарная длина пути.</returns>
        private double RouteLength(List<int> route)
        {
            double len = 0;
            for (int i = 0; i < route.Count - 1; i++)
                len += dist[route[i], route[i + 1]];
            return len;
        }

        /// <summary>
        /// Создаёт случайный маршрут, начиная со стартового города.
        /// Остальные города перемешиваются алгоритмом Фишера-Йетса.
        /// </summary>
        /// <returns>Новый случайный маршрут.</returns>
        private List<int> CreateRandomRoute()
        {
            var cities = Enumerable.Range(0, numCities).Where(c => c != startCity).ToList();

            // алгоритм Фишера-Йетса
            for (int i = cities.Count - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);
                (cities[i], cities[j]) = (cities[j], cities[i]);
            }

            var route = new List<int> { startCity };
            route.AddRange(cities);
            route.Add(startCity);
            return route;
        }

        /// <summary>
        /// Создаёт частично случайный маршрут:
        /// первые greedyNeighbors шагов выбираются как ближайшие непосещённые соседы,
        /// остальные – случайно.
        /// </summary>
        /// <returns>Новый случайный маршрут с greedyNeighbors жадными выборами.</returns>
        private List<int> CreateGreedyRoute()
        {
            var visited = new bool[numCities];
            var route = new List<int> { startCity };
            visited[startCity] = true;
            int current = startCity;

            var neighbors = new List<int>[numCities];
            for (int i = 0; i < numCities; i++)
            {
                neighbors[i] = [.. Enumerable.Range(0, numCities)
                    .Where(j => j != i)
                    .OrderBy(j => dist[i, j])];
            }

            int stepsGreedy = Math.Min(greedyNeighbors, numCities - 1);

            for (int step = 1; step < numCities; step++)
            {
                int next = -1;
                if (step <= stepsGreedy)
                {
                    // Жадный выбор
                    foreach (var city in neighbors[current])
                        if (!visited[city]) { next = city; break; }
                }
                // Если не нашли выбираем случайного непосещённого
                if (next == -1)
                {
                    var unvisited = Enumerable.Range(0, numCities).Where(c => !visited[c]).ToList();
                    next = unvisited[rand.Next(unvisited.Count)];
                }
                route.Add(next);
                visited[next] = true;
                current = next;
            }
            route.Add(startCity);
            return route;
        }

        /// <summary>
        /// Формирует начальную популяцию.
        /// Для каждой особи с вероятностью greedyProbability вызывается CreateGreedyRoute,
        /// иначе CreateRandomRoute.
        /// </summary>
        private void CreateInitialPopulation()
        {
            population.Clear();
            for (int i = 0; i < populationSize; i++)
            {
                List<int> route;
                if (rand.Next(100) < greedyProbability)
                    route = CreateGreedyRoute();
                else
                    route = CreateRandomRoute();
                population.Add(route);
            }
            fitness = population.Select(r => RouteLength(r)).ToArray();
        }

        /// <summary>
        /// Порядковый кроссовер.
        /// Первая половина потомка копируется от первого родителя,
        /// остальные позиции заполняются городами из второго родителя в порядке их следования без повторений.
        /// </summary>
        /// <param name="parent1">Первый родительский маршрут.</param>
        /// <param name="parent2">Второй родительский маршрут.</param>
        /// <returns>Маршрут потомка (без замыкающего города).</returns>
        private List<int> CrossoverOX(List<int> parent1, List<int> parent2)
        {
            int n = numCities;
            var p1 = parent1.Take(n).ToList();
            var p2 = parent2.Take(n).ToList();
            int half = n / 2;
            var child = new int[n];
            Array.Fill(child, -1);
            var inChild = new HashSet<int>();

            for (int i = 0; i < half; i++)
            {
                child[i] = p1[i];
                inChild.Add(p1[i]);
            }

            int pos = half;
            foreach (var city in p2)
            {
                if (!inChild.Contains(city))
                {
                    child[pos++] = city;
                    inChild.Add(city);
                }
            }
            return child.ToList();
        }

        /// <summary>
        /// Мутация: с вероятностью mutationChance меняет местами два случайных города в маршруте.
        /// </summary>
        /// <param name="route">Маршрут (без замыкающего города).</param>
        private void Mutate(List<int> route)
        {
            if (rand.NextDouble() > mutationChance) return;
            int i = rand.Next(numCities);
            int j = rand.Next(numCities);
            while (i == j) j = rand.Next(numCities);
            (route[i], route[j]) = (route[j], route[i]);
        }

        /// <summary>
        /// Отбирает случайную рабочую группу размера wrGroupSize, затем сортирует её по возрастанию длины.
        /// </summary>
        /// <returns>Список индексов маршрутов в популяции, отсортированный от лучшего к худшему.</returns>
        /// <remarks>
        /// Случайность помогает сохранять разнообразие и избегать преждевременной сходимости.
        /// </remarks>
        private List<int> SelectWorkingGroup()
        {
            var indices = Enumerable.Range(0, populationSize).ToList();
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int k = rand.Next(i + 1);
                (indices[i], indices[k]) = (indices[k], indices[i]);
            }
            var group = indices.Take(wrGroupSize).ToList();

            group.Sort((a, b) => fitness[a].CompareTo(fitness[b]));
            return group;
        }

        /// <summary>
        /// Обновляет глобально лучший маршрут, просматривая всю популяцию.
        /// </summary>
        private void UpdateBestRoute()
        {
            for (int i = 0; i < population.Count; i++)
            {
                if (fitness[i] < bestLength)
                {
                    bestLength = fitness[i];
                    bestRoute = new List<int>(population[i]);
                }
            }
        }

        /// <summary>
        /// Добавляет запись о текущем поколении в список generationRecords.
        /// </summary>
        /// <param name="generation">Номер поколения (начиная с 0).</param>
        private void SaveGenerationData(int generation)
        {
            var record = new
            {
                type = "generation",
                generation = generation + 1,
                bestLength = bestLength,
                bestRoute = new List<int>(bestRoute)
            };
            generationRecords.Add(record);
        }

        /// <summary>
        /// Запускает основной цикл генетического алгоритма на maxGenerations поколений.
        /// На каждом поколении: отбор рабочей группы, кроссовер, мутация, замена худшего,
        /// обновление лучшего, при необходимости сохранение данных в JSON.
        /// </summary>
        public void Run()
        {
            Console.WriteLine("\nЗапуск генетического алгоритма");

            // Добавление метаданных в начало JSON для корректной работы визуализатора
            if (!string.IsNullOrEmpty(outputJsonPath))
            {
                generationRecords.Add(new
                {
                    type = "metadata",
                    numCities = numCities,
                    startCity = startCity,
                    saveStep = saveStep,
                    coordinates = coords.Select(c => new { x = c.X, y = c.Y }).ToList(),
                    parameters = new
                    {
                        populationSize,
                        maxGenerations, // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        wrGroupSize,
                        mutationChance,
                        greedyNeighbors,
                        greedyProbability
                    }
                });
            }

            for (int gen = 0; gen < maxGenerations; gen++)
            {
                // 1. Отбор рабочей группы и получение родителей
                var wrGroupIndices = SelectWorkingGroup();

                // 2. Скрещивание двух лучших в рабочей группе
                var parent1 = population[wrGroupIndices[0]];
                var parent2 = population[wrGroupIndices[1]];
                var child = CrossoverOX(parent1, parent2);

                // 3. Мутация
                Mutate(child);

                // 4. Добавляем замыкающий город в конец
                child.Add(startCity);
                double childLen = RouteLength(child);

                // 5. Замена худшего маршрута в рабочей группе на потомка
                int worstIdx = wrGroupIndices[^1];
                population[worstIdx] = child;
                fitness[worstIdx] = childLen;

                // 6. Обновляем глобально лучший маршрут
                UpdateBestRoute();

                // 7. Сохраняем данные в JSON
                if (!string.IsNullOrEmpty(outputJsonPath) && (gen % saveStep == 0))
                    SaveGenerationData(gen);

                Console.WriteLine($"Поколение {gen + 1} / {maxGenerations}, лучшая длина = {bestLength:F2}");
            }

            if (!string.IsNullOrEmpty(outputJsonPath))
            {
                var options = new JsonSerializerOptions { WriteIndented = true };
                string json = JsonSerializer.Serialize(generationRecords, options);
                File.WriteAllText(outputJsonPath, json);
                Console.WriteLine($"\nДанные сохранены в {outputJsonPath}");
            }

            Console.WriteLine("\nАлгоритм завершён");
        }

        /// <summary>
        /// Выводит в консоль лучший найденный маршрут и его длину.
        /// </summary>
        public void PrintResult()
        {
            Console.WriteLine("\n==============================================");
            Console.WriteLine("Результат работы генетического алгоритма: ");
            Console.WriteLine($"Лучший найденный маршрут (длина = {bestLength:F2}):");
            Console.WriteLine(string.Join(" -> ", bestRoute));
            Console.WriteLine("==============================================");
        }
    }

    /// <summary>
    /// Содержит точку входа и все вспомогательные функции ввода/вывода,
    /// включая генерацию случайных городов, чтение/запись файлов и запуск Python-визуализации.
    /// </summary>
    static class Program
    {
        /// <summary>
        /// Главное меню программы. Предлагает 4 режима:
        /// ручной ввод, чтение из файла, генерация случайного файла, запуск визуализации по готовому JSON.
        /// </summary>
        static void Main()
        {
            Console.WriteLine("==============================================");
            Console.WriteLine("Генетический алгоритм для задачи коммивояжёра");
            Console.WriteLine("==============================================");

            Console.WriteLine("\nВыберите действие:");
            Console.WriteLine("1 - Ввод координат с клавиатуры");
            Console.WriteLine("2 - Чтение координат из файла");
            Console.WriteLine("3 - Сгенерировать случайные города и сохранить в файл");
            Console.WriteLine("4 - Запустить визуализацию (из существующего JSON файла)");
            Console.Write("Ваш выбор: ");
            string choiceStr = Console.ReadLine();
            int choice = int.Parse(choiceStr);

            if (choice == 4)
            {
                RunVisualizationOnly();
                return;
            }

            if (choice == 3)
            {
                GenerateRandomCitiesFile();
                Console.Write("\nЗапустить алгоритм с только что созданным файлом? (y/n): ");
                if (Console.ReadLine().ToLower() == "y")
                {
                    Console.Write("Введите имя файла: ");
                    string filename = Console.ReadLine();
                    if (ReadCoordinatesFromFile(filename, out int numCities, out (double X, double Y)[] coords))
                    {
                        int startCity = InputStartCity(numCities);
                        var (popSize, maxGen, wrGroupSize, mutChance, greedyNeighbors, greedyProb) = InputGAParameters(numCities);
                        var (jsonPath, saveStep) = AskForJsonSave();
                        var ga = new GeneticAlgorithm(numCities, coords, startCity,
                            popSize, maxGen, wrGroupSize, mutChance, greedyNeighbors, greedyProb,
                            jsonPath, saveStep);
                        ga.Run();
                        ga.PrintResult();
                        if (!string.IsNullOrEmpty(jsonPath))
                            OfferVisualization(jsonPath);
                    }
                }
                return;
            }

            // Режимы 1 и 2
            int citiesCount;
            (double X, double Y)[] coordsArr;

            if (choice == 2)
            {
                Console.Write("Введите имя файла: ");
                string filename = Console.ReadLine();
                if (!ReadCoordinatesFromFile(filename, out citiesCount, out coordsArr))
                    return;
                Console.WriteLine($"Данные загружены. Количество городов: {citiesCount}");
            }
            else if (choice == 1)
            {
                InputCitiesManual(out citiesCount, out coordsArr);
            }
            else
            {
                Console.WriteLine("Неверный выбор.");
                return;
            }

            int start = InputStartCity(citiesCount);
            var (popSize1, maxGen1, wrGroupSize1, mutChance1, greedyNeighbors1, greedyProb1) = InputGAParameters(citiesCount);
            var (jsonPath1, saveStep1) = AskForJsonSave();
            var gaObj = new GeneticAlgorithm(citiesCount, coordsArr, start,
                popSize1, maxGen1, wrGroupSize1, mutChance1, greedyNeighbors1, greedyProb1,
                jsonPath1, saveStep1);
            gaObj.Run();
            gaObj.PrintResult();

            if (!string.IsNullOrEmpty(jsonPath1))
                OfferVisualization(jsonPath1);
        }

        /// <summary>
        /// Предлагает пользователю запустить Python-визуализацию после завершения алгоритма.
        /// </summary>
        /// <param name="jsonPath">Путь к сохранённому JSON-файлу.</param>
        static void OfferVisualization(string jsonPath)
        {
            Console.Write("\nЗапустить визуализацию Python? (y/n): ");
            if (Console.ReadLine().ToLower() == "y")
                RunPythonVisualization(jsonPath);
        }

        /// <summary>
        /// Запускает визуализацию из существующего JSON-файла.
        /// </summary>
        static void RunVisualizationOnly()
        {
            Console.Write("Введите путь к JSON файлу (например, results.json): ");
            string jsonPath = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(jsonPath))
                jsonPath = "results.json";
            RunPythonVisualization(jsonPath);
        }

        /// <summary>
        /// Запускает внешний процесс Python с передачей ему имени JSON-файла.
        /// Ожидает завершения работы скрипта (пока пользователь не закроет окно matplotlib).
        /// </summary>
        /// <param name="jsonFilePath">Путь к JSON-файлу с историей поколений.</param>
        /// <exception cref="Exception">Ошибка запуска Python (например, python не в PATH).</exception>
        /// <remarks>
        /// Предполагается, что скрипт visualizer.py находится в той же папке, что и .exe.
        /// </remarks>
        static void RunPythonVisualization(string jsonFilePath)
        {
            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "visualizer.py");
            if (!File.Exists(jsonFilePath))
            {
                Console.WriteLine($"Ошибка: файл {jsonFilePath} не найден.");
                return;
            }

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{jsonFilePath}\"",
                UseShellExecute = true,
                CreateNoWindow = false,
                RedirectStandardOutput = false,
                RedirectStandardError = false
            };

            Console.WriteLine($"Запуск визуализации: python {scriptPath} {jsonFilePath}");
            try
            {
                using (var process = new Process { StartInfo = psi })
                {
                    process.Start();
                    process.WaitForExit();  // ждём, пока пользователь закроет окно с графикой
                    Console.WriteLine("Визуализация завершена.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Не удалось запустить Python: {ex.Message}");
                Console.WriteLine("Убедитесь, что Python установлен и доступен в PATH.");
            }
        }

        /// <summary>
        /// Запрашивает у пользователя, нужно ли сохранять историю в JSON,
        /// а также имя файла и шаг сохранения.
        /// </summary>
        /// <returns>Кортеж: путь к JSON-файлу (или null) и шаг сохранения.</returns>
        static (string path, int step) AskForJsonSave()
        {
            Console.Write("\nСохранять историю поколений в JSON файл? (y/n): ");
            if (Console.ReadLine().ToLower() == "y")
            {
                Console.Write("Имя JSON файла (например, results.json): ");
                string path = Console.ReadLine();
                Console.Write("Сохранять каждое N-е поколение (1 = все, 100 = каждое 100-е): ");
                int step = int.Parse(Console.ReadLine());
                if (step < 1) step = 1;
                return (path, step);
            }
            return (null, 1);
        }

        /// <summary>
        /// Генерирует текстовый файл со случайными координатами городов в заданном диапазоне.
        /// Формат: первая строка – количество городов, затем строки "x y".
        /// </summary>
        static void GenerateRandomCitiesFile()
        {
            Console.Write("Имя файла для сохранения: ");
            string filename = Console.ReadLine();
            Console.Write("Количество городов: ");
            int n = int.Parse(Console.ReadLine());
            Console.Write("Минимальная координата (по умолч. 0): ");
            int minCoord = int.Parse(Console.ReadLine());
            Console.Write("Максимальная координата (по умолч. 1000): ");
            int maxCoord = int.Parse(Console.ReadLine());

            using var writer = new StreamWriter(filename);
            writer.WriteLine(n);
            Random rand = Random.Shared;
            for (int i = 0; i < n; i++)
            {
                int x = rand.Next(minCoord, maxCoord + 1);
                int y = rand.Next(minCoord, maxCoord + 1);
                writer.WriteLine($"{x} {y}");
            }
            Console.WriteLine($"Файл {filename} создан с {n} городами.");
        }

        /// <summary>
        /// Чтение координат городов из текстового файла.
        /// </summary>
        /// <param name="filename">Имя файла.</param>
        /// <param name="numCities">Количество городов (выходной параметр).</param>
        /// <param name="coords">Массив координат (выходной параметр).</param>
        /// <returns>true, если чтение успешно, иначе false.</returns>
        static bool ReadCoordinatesFromFile(string filename, out int numCities, out (double X, double Y)[] coords)
        {
            numCities = 0;
            coords = null;
            try
            {
                using var reader = new StreamReader(filename);
                numCities = int.Parse(reader.ReadLine());
                coords = new (double, double)[numCities];
                for (int i = 0; i < numCities; i++)
                {
                    var parts = reader.ReadLine().Split();
                    double x = double.Parse(parts[0]);
                    double y = double.Parse(parts[1]);
                    coords[i] = (x, y);
                }
                return true;
            }
            catch
            {
                Console.WriteLine("Ошибка чтения файла.");
                return false;
            }
        }

        /// <summary>
        /// Ручной ввод координат с клавиатуры.
        /// </summary>
        /// <param name="numCities">Количество городов (выходной параметр).</param>
        /// <param name="coords">Массив координат (выходной параметр).</param>
        static void InputCitiesManual(out int numCities, out (double X, double Y)[] coords)
        {
            Console.Write("Количество городов: ");
            numCities = int.Parse(Console.ReadLine());
            coords = new (double, double)[numCities];
            Console.WriteLine("Введите координаты x y:");
            for (int i = 0; i < numCities; i++)
            {
                Console.Write($"Город {i}: ");
                var parts = Console.ReadLine().Split();
                coords[i] = (double.Parse(parts[0]), double.Parse(parts[1]));
            }
        }

        /// <summary>
        /// Запрашивает стартовый город и проверяет корректность индекса.
        /// </summary>
        /// <param name="numCities">Общее количество городов.</param>
        /// <returns>Корректный индекс стартового города.</returns>
        static int InputStartCity(int numCities)
        {
            Console.Write($"Стартовый город (0..{numCities - 1}): ");
            int start = int.Parse(Console.ReadLine());
            while (start < 0 || start >= numCities)
            {
                Console.Write($"Ошибка. Введите от 0 до {numCities - 1}: ");
                start = int.Parse(Console.ReadLine());
            }
            return start;
        }

        /// <summary>
        /// Ввод параметров генетического алгоритма с консоли.
        /// </summary>
        /// <param name="numCities">Количество городов (нужно для ограничения greedyNeighbors).</param>
        /// <returns>Кортеж с параметрами: размер популяции, поколения, размер рабочей группы,
        /// вероятность мутации, количество жадных соседей, вероятность жадности.</returns>
        static (int popSize, int maxGen, int wrGroupSize, double mutChance, int greedyNeighbors, int greedyProb) InputGAParameters(int numCities)
        {
            Console.WriteLine("\nВведите параметры генетического алгоритма:");
            Console.Write("Размер популяции: ");
            int popSize = int.Parse(Console.ReadLine());
            Console.Write("Максимальное количество поколений: ");
            int maxGen = int.Parse(Console.ReadLine());
            Console.Write("Размер рабочей группы: ");
            int wrGroupSize = int.Parse(Console.ReadLine());
            Console.Write("Вероятность мутации (0,0 ... 1,0): ");
            double mutChance = double.Parse(Console.ReadLine());
            Console.Write($"Количество ближайших соседей для жадного построения (макс. {numCities - 1}): ");
            int greedyNeighbors = int.Parse(Console.ReadLine());
            if (greedyNeighbors > numCities - 1) greedyNeighbors = numCities - 1;
            Console.Write("Вероятность использования жадного принципа (0..100): ");
            int greedyProb = int.Parse(Console.ReadLine());
            return (popSize, maxGen, wrGroupSize, mutChance, greedyNeighbors, greedyProb);
        }
    }
}