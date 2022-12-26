import random
import numpy as np
from PIL import Image
import os
import glob
import re


# Функция перевода изображения в вектор
def img_to_string(filename):
    img = Image.open(filename)
    img_as_array = np.asarray(img)
    s = ''
    for y in range(img_as_array.shape[0]):
        for x in range(img_as_array.shape[1]):
            if img_as_array[(y, x)].sum() >= 384:
                s = s + '0'
            else:
                s = s + '1'
    return s


# Функция сортировки словаря
def dict_reorder(dict, sort_part):
    list_nums = []
    for k, v in dict.items():
        match = re.search(sort_part, k)
        if match is not None:
            list_nums.append(int(match.group(1)))
    list_nums.sort()
    reordered_dict = {sort_part.replace("(\\d+)", str(n)): dict[sort_part.replace("(\\d+)", str(n))] for n in
                      list_nums}
    return reordered_dict


# Обучающая выборка
training_set_folder = './training_set'
objects = []
objects = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(rf'{training_set_folder}/*.jpg')]

# Тестовая выборка
test_set_folder = './test_set'
tests = {}
for test_path in glob.glob(rf'{test_set_folder}/*.jpg'):
    test_file = os.path.splitext(os.path.basename(test_path))[0]
    match = re.search("^(test[0-9]+)_(.+)$", test_file)
    if match is not None:
        tests[match.group(1)] = match.group(2)
tests = dict_reorder(tests, r"test(\d+)")

# Представление объектов
obj_representations = {}
for obj in objects:
    obj_representations[obj] = list(img_to_string(rf'{training_set_folder}/{obj}.jpg'))

# Количество входов
inputs_count = len(obj_representations[objects[0]])

# Инициализация весов нейронов сети
weights = {}
for obj in objects:
    weights[obj] = []
    for i in range(inputs_count):
        weights[obj].append(0)

# Порог функции активации (для всех нейронов)
bias = 7


# Активационная функция нейрона
def proceed(obj, inputs):
    # Взвешенная сумма
    net = 0
    for i in range(inputs_count):
        net += int(inputs[i]) * weights[obj][i]

    # Пороговая функция
    return net >= bias


# Правило № 1 из алгоритма Хэбба: увеличение значений весов, если сеть ошиблась и выдала 0
def increase(obj, number):
    for i in range(inputs_count):
        # Возбужденный ли вход
        if int(number[i]) == 1:
            # Увеличиваем связанный с ним вес на единицу
            weights[obj][i] += 1


# Правило №2 из алгоритма Хэбба: уменьшение значений весов, если сеть ошиблась и выдала 1
def decrease(obj, number):
    for i in range(inputs_count):
        # Возбужденный ли вход
        if int(number[i]) == 1:
            # Уменьшаем связанный с ним вес на единицу
            weights[obj][i] -= 1


# Обучение сети
for obj in objects:
    for i in range(10000):
        # Генерация случайного класса
        option = objects[random.randint(0, len(objects) - 1)]

        # Применение алгоритма Хэбба
        if option != obj:
            # Правило № 2
            if proceed(obj, obj_representations[option]):
                decrease(obj, obj_representations[option])
        else:
            # Правило № 1
            if not proceed(obj, obj_representations[option]):
                increase(obj, obj_representations[option])

print()

# Список классов
print('Список классов:')
print(objects)
print()

# Представления классов
print('Представления классов:')
print(obj_representations)
print()

# Значения весов
print('Вычисленные значения весов для каждого нейрона:')
print(weights)
print()

# Проход по тестовой выборке
print('Проход по тестовой выборке:')
for k, v in tests.items():
    print(rf"Тест {k} на определение {v}: результат: ",
          proceed(v, list(img_to_string(rf'{test_set_folder}/{k}_{v}.jpg'))))
print()
