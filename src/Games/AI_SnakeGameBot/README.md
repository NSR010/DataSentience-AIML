# Snake Game AI
**Author:** @abhi-ram-M 
---
<img width="500" height="400" alt="Screenshot 2025-10-09 180128" src="https://github.com/user-attachments/assets/5ddab6d5-2fd3-46a5-8322-d8394db97d9e" /><img width="300" height="300" alt="Screenshot 2025-10-09 180128" src="https://github.com/user-attachments/assets/49306284-4156-4608-bec1-2478a75cdb71" />
---

## 🎯 Overview
**AI SnakeMaster** is an AI-powered Snake Game built using **Reinforcement Learning** and **Deep Q-Learning (DQN)**.  
Instead of following a hardcoded or brute-force path, the Snake learns through **experience**, **reward feedback**, and **neural network training**.  

This project uses **PyTorch** for the deep learning backend and **Pygame** for game rendering.
---

## 🧠 Core Concept
- **Reinforcement Learning (RL):**  
  The agent (Snake) interacts with the environment (Game) and learns actions that maximize cumulative rewards.  
  - +10 → Eats food  
  - -10 → Game over  
  - 0 → Otherwise  

- **Deep Q-Learning (DQN):**  
  A neural network estimates the Q-values (expected future rewards) for each action.  
  The model improves by replaying past experiences and adjusting predictions over time.

---
## 🚀 Installation

### ⚠️ Important Note
> This project is compatible **only with Python 3.7 or 3.8**.  
> Some libraries used are **not supported** in newer Python versions (3.9+).  
> It is **highly recommended** to run this project inside a **fresh virtual environment**.
### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/abhi-ram-M/snake_game_ai.git
cd snake_game_ai
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# Or download python 3.8, add it to path and create a virtual env through vscode and select as python 3.8 as version and set it as default interpreter for the project.

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---
# ▶️ Quick Start
Train the AI Agent (Deep Q-Learning)
```bash
python agent.py

```
The agent starts with random actions and gradually learns smarter movement — avoiding walls and chasing food effectively.
---
# 🧩 Project Structure
```bash
snake_game_ai/
├── agent.py          # Reinforcement Learning logic
├── model.py          # Deep Q-Network (PyTorch)
├── snake_game.py     # Snake game environment (Pygame)
├── helper.py         # Graph plotter
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```
---
## ⚙️ Algorithm Summary

1. **Calculate the current state** — Determine the game’s status using 11 input parameters.  
2. **Predict the next action** — Use the trained neural network to decide the next move.  
3. **Execute the action** — Perform the action and observe the resulting reward.  
4. **Store experience** — Save `(state, action, reward, next_state)` in memory for learning.  
5. **Train the model** — Re-train using replayed experiences to improve decision-making.  
6. **Repeat** — Continue the cycle until the model’s performance consistently improves.

