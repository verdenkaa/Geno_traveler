#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <set>
#include <fstream>
#include <limits>

using namespace std;

// ================================ Класс для решения TSP генетическим алгоритмом ================================
class TSPGeneticAlgorithm {
private:
    int numCities;
    vector<pair<double, double>> coordinates;
    vector<vector<double>> dist;
    vector<vector<int>> closeCities;
    int startCity;

    int populationSize;
    int maxGenerations;
    int wrGroupSize;
    double mutationChance;
    int greedyNeighbors;
    int greedyProbability;

    vector<vector<int>> population;
    vector<double> fitness;
    vector<int> bestRoute;
    double bestLength;

    mt19937 rng;

    double euclideanDistance(const pair<double, double>& a, const pair<double, double>& b) {
        double dx = a.first - b.first;
        double dy = a.second - b.second;
        return sqrt(dx * dx + dy * dy);
    }

    void computeDistances() {
        dist.assign(numCities, vector<double>(numCities, 0.0));
        for (int i = 0; i < numCities; ++i) {
            for (int j = i + 1; j < numCities; ++j) {
                double d = euclideanDistance(coordinates[i], coordinates[j]);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
    }

    void computeCloseCities() {
        closeCities.resize(numCities);
        for (int i = 0; i < numCities; ++i) {
            vector<pair<double, int>> neighbors;
            for (int j = 0; j < numCities; ++j) {
                if (j != i) neighbors.emplace_back(dist[i][j], j);
            }
            sort(neighbors.begin(), neighbors.end());
            closeCities[i].reserve(numCities - 1);
            for (const auto& p : neighbors) closeCities[i].push_back(p.second);
        }
    }

    double computeRouteLength(const vector<int>& route) const {
        double len = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i) len += dist[route[i]][route[i + 1]];
        return len;
    }

    vector<int> createRandomRoute() {
        vector<int> cities;
        for (int i = 0; i < numCities; ++i) if (i != startCity) cities.push_back(i);
        shuffle(cities.begin(), cities.end(), rng);
        vector<int> route;
        route.push_back(startCity);
        route.insert(route.end(), cities.begin(), cities.end());
        route.push_back(startCity);
        return route;
    }

    vector<int> createGreedyRoute() {
        vector<bool> visited(numCities, false);
        vector<int> route;
        route.push_back(startCity);
        visited[startCity] = true;
        int current = startCity;
        int stepsGreedy = min(greedyNeighbors, numCities - 1);

        for (int step = 1; step < numCities; ++step) {
            if (step <= stepsGreedy) {
                bool found = false;
                for (int neighbor : closeCities[current]) {
                    if (!visited[neighbor]) {
                        route.push_back(neighbor);
                        visited[neighbor] = true;
                        current = neighbor;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    vector<int> unvisited;
                    for (int i = 0; i < numCities; ++i) if (!visited[i]) unvisited.push_back(i);
                    uniform_int_distribution<> distUnvisited(0, unvisited.size() - 1);
                    int nextCity = unvisited[distUnvisited(rng)];
                    route.push_back(nextCity);
                    visited[nextCity] = true;
                    current = nextCity;
                }
            }
            else {
                vector<int> unvisited;
                for (int i = 0; i < numCities; ++i) if (!visited[i]) unvisited.push_back(i);
                uniform_int_distribution<> distUnvisited(0, unvisited.size() - 1);
                int nextCity = unvisited[distUnvisited(rng)];
                route.push_back(nextCity);
                visited[nextCity] = true;
                current = nextCity;
            }
        }
        route.push_back(startCity);
        return route;
    }

    void createInitialPopulation() {
        population.clear();
        fitness.clear();
        uniform_int_distribution<> probDist(0, 99);

        for (int i = 0; i < populationSize; ++i) {
            int r = probDist(rng);
            vector<int> route = (r < greedyProbability) ? createGreedyRoute() : createRandomRoute();
            population.push_back(route);
            fitness.push_back(computeRouteLength(route));
        }
    }

    vector<int> crossoverOX(const vector<int>& parent1, const vector<int>& parent2) {
        int n = numCities;
        vector<int> p1(parent1.begin(), parent1.begin() + n);
        vector<int> p2(parent2.begin(), parent2.begin() + n);
        int half = n / 2;
        vector<int> child(n, -1);
        set<int> inChild;
        for (int i = 0; i < half; ++i) {
            child[i] = p1[i];
            inChild.insert(p1[i]);
        }
        int pos = half;
        for (int i = 0; i < n && pos < n; ++i) {
            int city = p2[i];
            if (inChild.find(city) == inChild.end()) {
                child[pos++] = city;
                inChild.insert(city);
            }
        }
        return child;
    }

    void mutate(vector<int>& route) {
        uniform_real_distribution<> prob(0.0, 1.0);
        if (prob(rng) <= mutationChance) {
            int n = numCities;
            uniform_int_distribution<> indexDist(0, n - 1);
            int i = indexDist(rng);
            int j = indexDist(rng);
            while (i == j) j = indexDist(rng);
            swap(route[i], route[j]);
        }
    }

    vector<int> selectWorkingGroup() {
        vector<int> indices(populationSize);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);
        indices.resize(wrGroupSize);
        sort(indices.begin(), indices.end(), [this](int a, int b) {
            return fitness[a] < fitness[b];
            });
        return indices;
    }

    void updateBestRoute() {
        for (size_t i = 0; i < population.size(); ++i) {
            if (fitness[i] < bestLength) {
                bestLength = fitness[i];
                bestRoute = population[i];
            }
        }
    }

public:
    TSPGeneticAlgorithm(int numCities, const vector<pair<double, double>>& coords,
        int startCity, int popSize, int maxGen, int wrGroupSize,
        double mutChance, int greedyNeighbors, int greedyProb)
        : numCities(numCities), coordinates(coords), startCity(startCity),
        populationSize(popSize), maxGenerations(maxGen), wrGroupSize(wrGroupSize),
        mutationChance(mutChance), greedyNeighbors(greedyNeighbors),
        greedyProbability(greedyProb), bestLength(numeric_limits<double>::max()) {
        random_device rd;
        rng = mt19937(rd());
        computeDistances();
        computeCloseCities();
        createInitialPopulation();
        updateBestRoute();
    }

    void run() {
        cout << "\n--- Запуск генетического алгоритма ---\n";
        for (int generation = 0; generation < maxGenerations; ++generation) {
            vector<int> wrGroupIndices = selectWorkingGroup();
            const vector<int>& parent1 = population[wrGroupIndices[0]];
            const vector<int>& parent2 = population[wrGroupIndices[1]];
            vector<int> childRoute = crossoverOX(parent1, parent2);
            mutate(childRoute);
            childRoute.push_back(startCity);
            double childLength = computeRouteLength(childRoute);
            int worstIdx = wrGroupIndices.back();
            population[worstIdx] = childRoute;
            fitness[worstIdx] = childLength;
            updateBestRoute();
            cout << "Поколение " << generation + 1 << " / " << maxGenerations
                << ", лучшая длина = " << bestLength << endl;
        }
        cout << "\n--- Алгоритм завершен ---\n";
    }

    void printResult() const {
        cout << "\n=== Результат работы генетического алгоритма ===\n";
        cout << "Лучший найденный маршрут (длина = " << bestLength << "):\n";
        for (size_t i = 0; i < bestRoute.size(); ++i) {
            cout << bestRoute[i];
            if (i < bestRoute.size() - 1) cout << " -> ";
        }
        cout << endl;
    }
};

// ================================ Функции ввода/вывода ================================
void generateRandomCitiesFile() {
    string filename;
    int numCities;
    int minCoord = 0, maxCoord = 1000;

    cout << "\n=== Генерация файла со случайными городами ===\n";
    cout << "Введите имя файла для сохранения: ";
    cin >> filename;
    cout << "Введите количество городов: ";
    cin >> numCities;
    if (numCities <= 0) {
        cout << "Ошибка: количество городов должно быть положительным.\n";
        return;
    }
    cout << "Введите минимальную координату (по умолчанию 0): ";
    cin >> minCoord;
    cout << "Введите максимальную координату (по умолчанию 1000): ";
    cin >> maxCoord;
    if (minCoord >= maxCoord) {
        cout << "Ошибка: minCoord должен быть меньше maxCoord. Используем 0 и 1000.\n";
        minCoord = 0;
        maxCoord = 1000;
    }

    ofstream file(filename);
    if (!file) {
        cout << "Ошибка: не удалось создать файл " << filename << endl;
        return;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> coordDist(minCoord, maxCoord);

    file << numCities << "\n";
    for (int i = 0; i < numCities; ++i) {
        file << coordDist(gen) << " " << coordDist(gen) << "\n";
    }
    file.close();

    cout << "Файл " << filename << " успешно создан с " << numCities << " городами.\n";
    cout << "Координаты лежат в диапазоне [" << minCoord << ", " << maxCoord << "].\n";
}

bool readCoordinatesFromFile(const string& filename, int& numCities, vector<pair<double, double>>& coords) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Ошибка: не удалось открыть файл " << filename << endl;
        return false;
    }
    file >> numCities;
    if (numCities <= 0) {
        cout << "Ошибка: некорректное количество городов в файле" << endl;
        return false;
    }
    coords.resize(numCities);
    for (int i = 0; i < numCities; ++i) {
        file >> coords[i].first >> coords[i].second;
        if (file.fail()) {
            cout << "Ошибка: неверный формат координат в файле" << endl;
            return false;
        }
    }
    file.close();
    return true;
}

void inputCitiesManual(int& numCities, vector<pair<double, double>>& coords) {
    cout << "Введите количество городов: ";
    cin >> numCities;
    coords.resize(numCities);
    cout << "Введите координаты городов (x y):\n";
    for (int i = 0; i < numCities; ++i) {
        cout << "Город " << i << ": ";
        cin >> coords[i].first >> coords[i].second;
    }
}

int inputStartCity(int numCities) {
    int start;
    cout << "Введите стартовый город (0 .. " << numCities - 1 << "): ";
    cin >> start;
    while (start < 0 || start >= numCities) {
        cout << "Ошибка: город должен быть в диапазоне 0.." << numCities - 1 << ". Повторите: ";
        cin >> start;
    }
    return start;
}

void inputGAParameters(int& popSize, int& maxGen, int& wrGroupSize, double& mutChance,
    int& greedyNeighbors, int& greedyProb, int numCities) {
    cout << "\nВведите параметры генетического алгоритма:\n";
    cout << "Размер популяции: ";
    cin >> popSize;
    cout << "Максимальное количество поколений: ";
    cin >> maxGen;
    cout << "Размер рабочей группы (должен быть <= размеру популяции): ";
    cin >> wrGroupSize;
    while (wrGroupSize > popSize) {
        cout << "Ошибка: размер рабочей группы не может превышать размер популяции. Повторите: ";
        cin >> wrGroupSize;
    }
    cout << "Вероятность мутации (0..1): ";
    cin >> mutChance;
    cout << "Количество ближайших соседей для жадного построения (макс. " << numCities - 1 << "): ";
    cin >> greedyNeighbors;
    if (greedyNeighbors > numCities - 1) greedyNeighbors = numCities - 1;
    cout << "Вероятность использования жадного принципа при создании начальной популяции (0..100): ";
    cin >> greedyProb;
}

// ================================ Главная функция ================================
int main() {
    setlocale(LC_ALL, "");

    cout << "==========================================\n";
    cout << "Генетический алгоритм для задачи коммивояжера\n";
    cout << "==========================================\n";

    int choice;
    cout << "\nВыберите действие:\n";
    cout << "1 - Ввод координат с клавиатуры\n";
    cout << "2 - Чтение координат из файла\n";
    cout << "3 - Сгенерировать случайные города и сохранить в файл\n";
    cout << "Ваш выбор: ";
    cin >> choice;

    if (choice == 3) {
        generateRandomCitiesFile();
        // После генерации файла спросим, хочет ли пользователь сразу использовать его
        char runNow;
        cout << "\nЗапустить алгоритм с только что созданным файлом? (y/n): ";
        cin >> runNow;
        if (runNow == 'y' || runNow == 'Y') {
            // Здесь нужно повторно прочитать параметры и запустить ГА
            string filename;
            cout << "Введите имя файла (то же, что указали при генерации): ";
            cin >> filename;
            int numCities;
            vector<pair<double, double>> coords;
            if (!readCoordinatesFromFile(filename, numCities, coords)) {
                return 1;
            }
            int startCity = inputStartCity(numCities);
            int popSize, maxGen, wrGroupSize, greedyNeighbors, greedyProb;
            double mutChance;
            inputGAParameters(popSize, maxGen, wrGroupSize, mutChance, greedyNeighbors, greedyProb, numCities);
            TSPGeneticAlgorithm ga(numCities, coords, startCity, popSize, maxGen,
                wrGroupSize, mutChance, greedyNeighbors, greedyProb);
            ga.run();
            ga.printResult();
        }
        else {
            cout << "Выход. Запустите программу снова и выберите пункт 2 для использования файла.\n";
        }
        return 0;
    }

    // Режимы 1 и 2 (ручной ввод или из файла)
    int numCities;
    vector<pair<double, double>> coords;

    if (choice == 2) {
        string filename;
        cout << "Введите имя файла: ";
        cin >> filename;
        if (!readCoordinatesFromFile(filename, numCities, coords)) {
            return 1;
        }
        cout << "Данные успешно загружены из файла. Количество городов: " << numCities << endl;
    }
    else if (choice == 1) {
        inputCitiesManual(numCities, coords);
    }
    else {
        cout << "Неверный выбор. Завершение работы.\n";
        return 1;
    }

    int startCity = inputStartCity(numCities);
    int popSize, maxGen, wrGroupSize, greedyNeighbors, greedyProb;
    double mutChance;
    inputGAParameters(popSize, maxGen, wrGroupSize, mutChance, greedyNeighbors, greedyProb, numCities);

    TSPGeneticAlgorithm ga(numCities, coords, startCity, popSize, maxGen,
        wrGroupSize, mutChance, greedyNeighbors, greedyProb);
    ga.run();
    ga.printResult();

    return 0;
}