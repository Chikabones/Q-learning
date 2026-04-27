import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Q-Learning Maze", layout="centered")

st.header("🎬 Работа обученного ИИ")
st.image("12-40-25 (online-video-cutter.com).gif", caption="Агент ищет путь")

st.markdown("""
### Chikabones:
**1. Что делает ИИ?**
Синий шар — это наш агент. Его цель: найти выход **(H)**, собрать золото **(G)** и не попасться монстру **(M)**.

**2. Как он учится?**
ИИ не знает правил заранее. Он обучается методом проб и ошибок в течение 2000 эпизодов:

* **За каждый шаг:** получает -1 (штраф за время).
* **За стену:** получает -10 (удар).
* **За золото:** получает +50 (бонус).
* **За монстра:** получает -1000 (смерть).
* **За финиш:** получает +1000 (победа).

**Экшен (Action):** Это выбор направления (Вверх, Вниз, Влево, Вправо).
""")

st.divider()

MAP = [
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", " ", " ", "#", " ", " ", " ", " ", "G", "#"],
    ["#", " ", " ", "#", " ", "#", "#", "#", " ", "#"],
    ["#", " ", " ", " ", " ", " ", " ", "#", " ", "#"],
    ["#", "#", "#", "#", "#", " ", " ", "#", " ", "#"],
    ["#", "G", " ", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", "#", "#", "#", "#", "#", "#", "M", "#"],
    ["#", " ", "#", " ", " ", " ", " ", " ", " ", "#"],
    ["#", " ", " ", " ", " ", " ", "#", " ", "H", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#"]
]

if 'q_table' not in st.session_state:
    st.session_state.q_table = {}

def draw_maze(r, c):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    for ir, row in enumerate(MAP):
        for ic, col in enumerate(row):
            y = 9 - ir
            if col == "#": ax.add_patch(patches.Rectangle((ic, y), 1, 1, color='black'))
            elif col == "H": ax.text(ic+0.5, y+0.5, "H", color='red', weight='bold', ha='center', va='center', fontsize=15)
            elif col == "G": ax.add_patch(patches.Circle((ic+0.5, y+0.5), 0.2, color='gold'))
            elif col == "M": ax.text(ic+0.5, y+0.5, "M", color='black', weight='bold', ha='center', va='center', fontsize=15)
    ax.add_patch(patches.Circle((c+0.5, 9-r+0.5), 0.3, color='blue', ec='white'))
    return fig

def train():
    q = st.session_state.q_table
    history = []
    for ep in range(2000):
        r, c, total_rew = 1, 1, 0
        for _ in range(100):
            s = (r, c)
            if s not in q: q[s] = np.zeros(4)
            a = np.argmax(q[s]) if random.random() > 0.1 else random.randint(0, 3)
            dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][a]
            nr, nc = r + dr, c + dc
            if nr<0 or nr>=10 or nc<0 or nc>=10 or MAP[nr][nc]=="#":
                rew, nr, nc = -10, r, c
            elif (nr, nc) == (8, 8): rew = 1000
            elif (nr, nc) == (6, 8): rew = -1000
            elif MAP[nr][nc] == "G": rew = 50
            else: rew = -1
            total_rew += rew
            ns = (nr, nc)
            if ns not in q: q[ns] = np.zeros(4)
            q[s][a] += 0.5 * (rew + 0.9 * np.max(q[ns]) - q[s][a])
            r, c = nr, nc
            if rew in [1000, -1000]: break
        history.append(total_rew)
    return history

if st.button("Запустить обучение"):
    rewards = train()
    
    st.subheader("📊 График обучения")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.plot(rewards, color='green')
    ax_hist.set_xlabel("Попытки"); ax_hist.set_ylabel("Баллы")
    st.pyplot(fig_hist)
    plt.close(fig_hist)
    
    st.subheader("🏁 Финальный проход")
    place = st.empty()
    r, c = 1, 1
    for _ in range(60):
        with place.container():
            current_fig = draw_maze(r, c)
            st.pyplot(current_fig)
            plt.close(current_fig)
        
        if (r, c) == (8, 8): 
            st.success("ИИ дошел до цели (H)!"); break
        if (r, c) == (6, 8): 
            st.error("ИИ попал в ловушку (M)!"); break
            
        a = np.argmax(st.session_state.q_table.get((r, c), np.zeros(4)))
        dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][a]
        r, c = r + dr, c + dc
        time.sleep(0.1)