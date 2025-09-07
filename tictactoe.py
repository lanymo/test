import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Tic-Tac-Toe 환경 정의
class TicTacToeEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.board.flatten()
    
    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0 or self.done:
            return self.board.flatten(), -10, True  # Invalid move penalty
        
        self.board[row, col] = self.current_player
        reward, self.done, self.winner = self.check_game_status()
        self.current_player *= -1
        return self.board.flatten(), reward, self.done
    
    def check_game_status(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return (1 if self.current_player == 1 else -1), True, self.current_player
        if abs(sum(np.diag(self.board))) == 3 or abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return (1 if self.current_player == 1 else -1), True, self.current_player
        if not (self.board == 0).any():
            return 0, True, 0  # Tie
        return 0, False, None
    
    def get_valid_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN 에이전트
class DQNAgent:
    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95, lr=0.001):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = deque(maxlen=2000)
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        q_values = q_values.detach().numpy().flatten()
        valid_q_values = {a: q_values[a] for a in valid_actions}
        return max(valid_q_values, key=valid_q_values.get)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            q_values = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f = q_values.clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 학습 루프
env = TicTacToeEnv()
agent1 = DQNAgent()
agent2 = DQNAgent()

num_episodes = 3000
batch_size = 32
update_target_freq = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent1.act(state, valid_actions) if env.current_player == 1 else agent2.act(state, valid_actions)
        next_state, reward, done = env.step(action)
        agent1.remember(state, action, reward if env.current_player == -1 else -reward, next_state, done)
        agent2.remember(state, action, reward if env.current_player == 1 else -reward, next_state, done)
        state = next_state
    
    agent1.replay(batch_size)
    agent2.replay(batch_size)
    
    if episode % update_target_freq == 0:
        agent1.update_target()
        agent2.update_target()
        print(f"Episode {episode}: Target network updated")

print("Training complete!")
