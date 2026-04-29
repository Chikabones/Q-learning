# 🤖 Q-Learning Maze AI

Обучение ИИ (Reinforcement Learning) прохождению лабиринта.

## 🚀 Запуск
1. Установи библиотеки:
   ```bash
   pip install -r requirements.txt
Запусти проект:

Bash
streamlit run train.py
🎮 Правила
Синий шар — Агент.

+2000 — Победа (дошел до дома H).

+300 — Сбор золота (G).

-1000 — Проигрыш (попал к монстру M).

-15 — Удар об стену (#).

-2 — Штраф за каждый шаг (чтобы ИИ не стоял на месте).

🛠 Технологии
Python, Streamlit, NumPy, Matplotlib.
