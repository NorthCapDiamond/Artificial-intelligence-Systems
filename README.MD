# Отчёт по модулю №1

## Введение:

### Описание целей проекта и его значимости.

Целью моего проекта стала разработка системы поддержки принятия решения на основе запросов F1 на основе онтологии, разработанной в [лабораторной работе 2](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab2), которая, строилась на основе базы знаний из [лабораторной работы 1](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab1).

## Анализ требований:

### Определение основных требований к системе поддержки принятия решений.

- Получение информации о конкретном Racer, Racing Team, Event, Race и связях между ними (и их instances)
- Простота в использовании (система предназначена для людей с разными знаниями в области F1 races)
- Множественность вывода (в зависимости от введенных пользователем данных, система должна менять реакцию)
- Надежность. Необходимо обработать некорректно введенные данные пользователя

### Выявление требований к базе знаний и онтологии для представления знаний.
- Создание Classes: Racer, Racing Team, Event, Race
- Заполнение классов корректными instances
- Создание связей между ними (Object properties)
- Создание правил для управления данными 
- создание тестовых запросов

## Изучение основных концепций и инструментов:

### Обзор основных концепций баз знаний и онтологий.

Базы знаний - это системы, которые хранят информацию и знания, которые могут быть использованы для решения задач. Они обычно содержат факты, правила, процедуры и другие элементы, необходимые для решения задач. Базы знаний используются в различных областях, таких как медицина, финансы, право и т.д.

Онтологии - это формальные модели, которые описывают концептуальные категории и отношения между ними. Они используются для описания знаний в языке, который может быть понят людьми и машинами. Онтологии используются для создания более точных и эффективных систем поиска, классификации и анализа данных.

Основными принципами баз знаний являются:

- Декомпозиция знаний на факты, правила и процедуры.
- Построение моделей знаний на основе логических и математических принципов.

Основными принципами онтологий являются:

- Описание концептуальных категорий и отношений между ними.
- Использование формальных языков для описания знаний.
- Создание стандартных моделей для обмена и интеграции данных.


### Изучение Prolog и его возможностей для разработки систем искусственного интеллекта.

1) Принципы и концепции
   - Логическое программирование (факты, правила, поиск решений на основе утверждений)
   - Сопоставление значений и термов для поиска решения задачи
   - Оптимизация (ускорение работы программы путем отбрасывания лишних вариантов; бывают красные и зелёные в зависимости от того, влияют они на логику программы или нет)
   - Рекурсия (поддержка рекурсивных вызовов)
2) Синтаксис
   - Предикаты (описание отношений)
   - Переменные (искомые значения)
   - Константы (числа, атомы, строки)
   - Операторы (арифметические, логические, сравнения)
3) Решение задач искусственного интеллекта
   - Формализация знаний и автоматическое принятие решений
   - Доказательство теорем
   - Анализ естественного языка
4) Функциональные возможности
   - Работа с БЗ (запросы, поиск решений)
   - Логический вывод (вывод новых фактов на основе имеющихся)
   - Обработка естественного языка (полезен для систем обработки текстов и диалогов)
5) Преимущества
   - Декларативность
   - Удобство
   - Решение задач ичкусственного интеллекта
6) Недостатки
   - Ограничен на больших наборах данных
   - Сложность в отладке
7) Актуальность
   - Язык считается мертвым

### Ознакомление с инструментами и библиотеками, подходящими для работы с базами знаний и онтологиями на Prolog.
Примеры: OWL, PL/SQL, SQLAlchemy, TensorFlow, MongoDB, ...
## Реализация системы искусственного интеллекта на Prolog:

Примеры:
- Машинное обучение
- Обработка естественного языка
- Составление поддержки принятия решения

Реализация же мой системы доступна [lab3](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab3).


Примеры запросов в БЗ (это тесты) и онтологию продемонстрированы в соответствующих отчетах к [лабораторной №1](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab1) и [лабораторной №2](https://github.com/NorthCapDiamond/Artificial-intelligence-Systems/tree/main/lab2).

Сравнение БЗ и Онтологии  их:

|     Что       |  Реализация в БЗ   |                   Реализация в онтологии                    |
|:-------------:|:------------------:|:-----------------------------------------------------------:|
|     Факт      |      Переменная    |                    Количество элементов                     |
|     Связь     |      Правило       |                     Наличие properties                      |
|   Рекурсия    |      Итерации      |          Reasoner обеспечивает установление связей          |


### Оценка и интерпретация результатов:

prolog:

is_winner_of_race("BAHRAIN GP", "Max Verstappen").

is_winner_of_race("BAHRAIN GP", "Dmitry Drobysh").

is_second_in_race("BAHRAIN GP", "Max Verstappen").

is_second_in_race("BAHRAIN GP", "Sergio Pérez").

is_third_in_race("BAHRAIN GP", "Fernando Alonso").

is_third_in_race("BAHRAIN GP", "Max Verstappen").


is_race_winner_in_team("BAHRAIN GP", "Red Bull").

is_race_second_in_team("BAHRAIN GP", "Red Bull").

is_race_second_in_team("BAHRAIN GP", "Ferrari").

is_race_third_in_team("BAHRAIN GP", "McLauren").

is_race_third_in_team("BAHRAIN GP", "Ferrari").

is_race_third_in_team("BAHRAIN GP", "Alpine").

is_race_third_in_team("BAHRAIN GP", "Aston Martin").



is_team_on_podium("BAHRAIN GP", "Red Bull").

is_team_on_podium("BAHRAIN GP", "Alpine").


protege: 


% Rule0 Return All parameters
Event

Race

Racer

Racing_Team




% Rule1 Show Racers From Team

Be_In_Team value Red_Bull




% Rule2 Return The Winner/Second/Third Of The Race
Win_The_Race value BAHRAIN_GP

Be_The_Second_In_Race value AUSTRALIAN_GP

Be_The_Third_In_Race value SAUDI_ARABIAN_GP




% Rule3 Return event that was in race
Event_In_Race value BAHRAIN_GP




% Rule4 Return Team Of A Winner/Second/Third
inverse(Be_In_Team) some (Win_The_Race value BAHRAIN_GP)

inverse(Be_In_Team) some (Be_The_Second_In_Race value AUSTRALIAN_GP)

inverse(Be_In_Team) some (Be_The_Third_In_Race value SAUDI_ARABIAN_GP)




% Rule5 Return Teams on podium
inverse(Be_In_Team) some (Be_On_Podium value AUSTRALIAN_GP)




% Rule6 see the racers that DNF
DNF value BAHRAIN_GP




% Rule7 see teams, where one or both pilots DNF
inverse(Be_In_Team) some (DNF value BAHRAIN_GP)




% Rule8 See The Partner of a Racer
Be_A_Partner value Max_Verstappen




% Rule9 See team if both drivers are on podium
((inverse (Be_In_Team) some (Win_The_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP))) or
((inverse (Be_In_Team) some (Win_The_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Third_In_Race value BAHRAIN_GP))) or
((inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP)) and (inverse (Be_In_Team) some (Be_The_Second_In_Race value BAHRAIN_GP)))




% Rule10 See team if one driver on podium and another one DNF
((inverse (Be_In_Team) some (Be_On_Podium value SAUDI_ARABIAN_GP)) and (inverse (Be_In_Team) some (DNF value SAUDI_ARABIAN_GP)))


SPRQL: 

// info param + 

// show racers from team param +

// i like to see the team that dnf  +

// i like to see the racer that dnf +

// i hate to see the team that dnf  +

// i hate to see the racer that dnf +

// i like to see the winner of param +

// i hate to see the winner of param +

// i like param racers 

// i hate param racers 



### Оценка соответствия системы поставленным требованиям и достижению целей проекта.

!!!Это решать моему практику!!!

### Интерпретация результатов и описание дальнейших возможностей развития и улучшения системы.

Полученные результаты можно интерпретировать как ИИ-помощника, действующего на основе онтологии, способной советовать пользователю команд и гонщиков из F1

## Заключение:

### Описание преимуществ и потенциальных применений разработанной системы искусственного интеллекта на базе Prolog, Protege, Python3.

Система уже может использоваться.


