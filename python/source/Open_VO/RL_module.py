from typing import List, Union, Deque, Tuple
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

from .data_types import MemoryCell, PPOMemoryCell, ActionResult


class BaseModelDQN(nn.Module):
    """
    Base DQN NN model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(BaseModelDQN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelPG(nn.Module):
    """
    Base PG NN model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(BaseModelPG, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelSARSA(nn.Module):
    """
    Base SARSA NN model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(BaseModelSARSA, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelA2CActor(nn.Module):
    """
    Base A2C NN actor model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(BaseModelA2CActor, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelA2CValue(nn.Module):
    """
    Base A2C NN Value model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super(BaseModelA2CValue, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, 1))  # Output is a single value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelPPOActor(nn.Module):
    """
    Base PPO NN actor model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(BaseModelPPOActor, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseModelPPOValue(nn.Module):
    """
    Base PPO NN value model class
    """
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super(BaseModelPPOValue, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        self.fc_layers = nn.ModuleList()
        prev_layer_size = self.input_size
        for i, layer_size in enumerate(self.hidden_sizes):
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.fc_layers.append(nn.Linear(prev_layer_size, 1))  # Output is a single value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x


class BaseTrainerDQN:
    """
    Base DQN trainer class
    """
    def __init__(self, model: BaseModelDQN, gamma: float, optimizer_adam: optim.Adam, device: Union[torch.device, str]):
        self.model = model
        self.gamma = gamma
        self.optimizer_adam = optimizer_adam
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model.to(self.device)

    def train_step(self, state: List[List[int]], action: List[List[int]],
                   reward: List[List[float]], next_state: List[List[int]], done: List[bool]):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # If only one parameter to train, then convert to a tuple of shape (1, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0).to(self.device)
            next_state = torch.unsqueeze(next_state, 0).to(self.device)
            action = torch.unsqueeze(action, 0).to(self.device)
            reward = torch.unsqueeze(reward, 0).to(self.device)
            done = (done,)

        # Predicted Q value with the current state
        # DQN: Q_new = reward + gamma * max(next_predicted Qvalue)
        pred = self.model(state).to(self.device)
        target = pred.clone().to(self.device)
        for idx in range(len(done)):
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).to(
                    self.device)
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer_adam.zero_grad()
        loss = F.mse_loss(target, pred)
        loss.backward()
        self.optimizer_adam.step()


class BaseTrainerPG:
    """
    Base PG trainer class
    """
    def __init__(self, model: BaseModelPG, optimizer_adam: optim.Adam, device: Union[torch.device, str]):
        self.model = model
        self.optimizer_adam = optimizer_adam
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model.to(self.device)

    def train_step(self, state: List[List[int]], action: List[List[int]],
                   reward: List[List[float]]):  # reward is cumulative rewards (returns)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        cum_reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # Predicted next expected action with the current state based on stochastic space
        # PG: expectation = integral(cumulative_reward * distribution(a | policy(s)))
        self.optimizer_adam.zero_grad()
        logits = self.model(state).to(self.device)
        log_probs = -F.cross_entropy(logits, action, reduction="none")
        loss = -log_probs * cum_reward
        loss.sum().backward()
        self.optimizer_adam.step()


class BaseTrainerSARSA:
    """
    Base SARSA trainer class
    """
    def __init__(self, model: BaseModelSARSA, gamma: float, optimizer_adam: optim.Adam,
                 device: Union[torch.device, str]):
        self.model = model
        self.gamma = gamma
        self.optimizer_adam = optimizer_adam
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model.to(self.device)

    def train_step(self, state: List[List[int]], action: List[List[int]],
                   reward: List[List[float]], next_state: List[List[int]],
                   next_action: List[List[int]], done: List[bool]):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        next_action = torch.tensor(next_action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # If only one parameter to train, then convert to a tuple of shape (1, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0).to(self.device)
            next_state = torch.unsqueeze(next_state, 0).to(self.device)
            action = torch.unsqueeze(action, 0).to(self.device)
            next_action = torch.unsqueeze(next_action, 0).to(self.device)
            reward = torch.unsqueeze(reward, 0).to(self.device)
            done = (done,)

        # Predicted Q value with the current state
        # SARSA: Q_new = reward + gamma * (next_predicted Qvalue)[argmax(next_action)]
        # next_action is getting from the
        # 1. act: taking an epsilon into account
        # OR
        # 2. net
        pred = self.model(state).to(self.device)
        target = pred.clone().to(self.device)
        for idx in range(len(done)):
            if not done[idx]:
                Q_new = reward[idx] + self.gamma \
                        * self.model(next_state[idx])[torch.argmax(next_action).item()].to(self.device)
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer_adam.zero_grad()
        loss = F.mse_loss(target, pred)
        loss.backward()
        self.optimizer_adam.step()


class BaseTrainerA2C:
    """
    Base A2C trainer class
    """
    def __init__(self, actor_model: BaseModelA2CActor, value_model: BaseModelA2CValue,
                 optimizer_adam_actor: optim.Adam, optimizer_adam_value: optim.Adam,
                 device: Union[torch.device, str]):
        self.actor_model = actor_model
        self.value_model = value_model
        self.optimizer_adam_actor = optimizer_adam_actor
        self.optimizer_adam_value = optimizer_adam_value
        self.device = torch.device(device) if isinstance(device, str) else device

        self.actor_model.to(self.device)
        self.value_model.to(self.device)

    def train_step(self, state: List[List[int]], action: List[List[int]],
                   reward: List[List[float]]):  # reward is cumulative rewards (returns G_t)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        cum_reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # Predicted next expected action with the current state based on stochastic space and value function
        # A2C: expectation = integral(cumulative_reward * distribution(a | policy(s))) - advantage
        self.optimizer_adam_value.zero_grad()
        values = self.value_model(state).to(self.device)
        values = values.squeeze(dim=1)
        value_loss = F.mse_loss(values, cum_reward)
        value_loss.sum().backward()
        self.optimizer_adam_value.step()

        with torch.no_grad():
            values = self.value_model(state).to(self.device)
        self.optimizer_adam_actor.zero_grad()
        advantage = cum_reward - values
        logits = self.actor_model(state).to(self.device)
        log_probs = -F.cross_entropy(logits, action, reduction="none")
        pi_loss = -log_probs * advantage
        pi_loss.sum().backward()
        self.optimizer_adam_actor.step()


class BaseTrainerPPO:
    """
    Base PPO trainer class
    """
    def __init__(self, actor_model: BaseModelPPOActor, value_model: BaseModelPPOValue,
                 kl_coeff: float, v_coeff: float,
                 optimizer_adam: optim.Adam,  # Single optimizer for both actor and value parameters
                 device: Union[torch.device, str]):
        self.actor_model = actor_model
        self.value_model = value_model
        self.kl_coeff = kl_coeff
        self.v_coeff = v_coeff
        self.optimizer_adam = optimizer_adam  # Assumes joint optimization or separate optimizers passed combined
        self.device = torch.device(device) if isinstance(device, str) else device

        self.actor_model.to(self.device)
        self.value_model.to(self.device)

    def train_step(self, state: List[List[int]], action: List[List[int]],
                   logits_old: List[List[float]], log_probs_old: List[List[float]],  # log_probs_old is [N, 1]
                   reward: List[List[float]]):  # reward is cumulative (returns G_t)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        cum_reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        logits_old = torch.tensor(logits_old, dtype=torch.float).to(self.device)
        log_probs = torch.tensor(log_probs_old, dtype=torch.float).to(self.device)
        log_probs = log_probs.unsqueeze(dim=1)

        # Predicted next expected action with the current state based on stochastic space and value function (A2C),
        # with penalty for preventing large policy update
        # PPO: expectation = A2C - penalty
        self.optimizer_adam.zero_grad()
        values_new = self.value_model(state).to(self.device)
        logits_new = self.actor_model(state).to(self.device)
        advantage = cum_reward - values_new

        log_probs_new = -F.cross_entropy(logits_new, action, reduction="none")
        log_probs_new = log_probs_new.unsqueeze(dim=1)
        prob_ratio = torch.exp(log_probs_new - log_probs)

        l0 = logits_old - torch.amax(logits_old, dim=1, keepdim=True)  # Reduce quantity
        l1 = logits_new - torch.amax(logits_new, dim=1, keepdim=True)  # Reduce quantity
        e0 = torch.exp(l0)
        e1 = torch.exp(l1)
        e_sum0 = torch.sum(e0, dim=1, keepdim=True)
        e_sum1 = torch.sum(e1, dim=1, keepdim=True)
        p0 = e0 / e_sum0
        kl = torch.sum(p0 * (l0 - torch.log(e_sum0) - l1 + torch.log(e_sum1)), dim=1, keepdim=True)

        value_loss = F.mse_loss(values_new, cum_reward)
        loss = -advantage * prob_ratio + kl * self.kl_coeff + value_loss * self.v_coeff
        loss.sum().backward()
        self.optimizer_adam.step()


class BaseAgentDQN:
    """
    Base DQN agent class
    """
    def __init__(self, max_memory: int, batch_size: int, rand_coef: int, rand_range: int,
                 model: BaseModelDQN, trainer: BaseTrainerDQN):
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.rand_coef = rand_coef  # for epsilon calculation
        self.rand_range = rand_range  # for epsilon calculation range
        self.model = model
        self.trainer = trainer  # Trainer already has device info

        self.n_episode = 0
        self.epsilon = 0  # Will be updated in act()
        self.memory: Deque[MemoryCell] = deque(maxlen=self.max_memory)

        self.model.to(self.trainer.device)

    def remember(self, memory_cell: MemoryCell):
        self.memory.append(memory_cell)

    def train_short_memory(self, memory_cell: MemoryCell):
        state = [memory_cell.state]
        action = [memory_cell.action]
        next_state = [memory_cell.next_state] if memory_cell.next_state is not None else [[]]  # Handle None case
        reward = [[memory_cell.reward]]
        done = [memory_cell.done if memory_cell.done is not None else False]

        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(list(self.memory), self.batch_size)
        else:
            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for cell in mini_sample:
            states.append(cell.state)
            actions.append(cell.action)
            rewards.append([cell.reward])
            next_states.append(cell.next_state if cell.next_state is not None else [])  # Handle None case
            dones.append(cell.done if cell.done is not None else False)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def act(self, state: List[int]) -> List[int]:
        self.epsilon = self.rand_coef - self.n_episode
        final_move = [0] * self.model.num_classes

        if random.randint(0, self.rand_range) < self.epsilon:
            move_idx = random.randint(0, self.model.num_classes - 1)
            final_move[move_idx] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.trainer.device)
            with torch.no_grad():
                prediction = self.model(state_t)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move


class BaseAgentPG:
    """
    Base PG agent class
    """
    def __init__(self, model: BaseModelPG, trainer: BaseTrainerPG):
        self.model = model
        self.trainer = trainer
        self.n_episode = 0

        self.model.to(self.trainer.device)

    def train_episode(self, memory: List[MemoryCell]):
        if not memory:
            return

        states, actions, rewards = [], [], []
        for cell in memory:
            states.append(cell.state)
            actions.append(cell.action)
            rewards.append([cell.reward])  # Trainer expects List[List[float]] for rewards

        self.trainer.train_step(states, actions, rewards)

    def act(self, state: List[int]) -> List[int]:
        final_move = [0] * self.model.num_classes

        state_t = torch.tensor(state, dtype=torch.float32, device=self.trainer.device)
        with torch.no_grad():
            logits = self.model(state_t).squeeze(0)  # Remove batch dim for multinomial

        probs = F.softmax(logits, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()
        final_move[move_idx] = 1

        return final_move


class BaseAgentSARSA:
    """
    Base SARSA agent class
    """
    def __init__(self, max_memory: int, batch_size: int, rand_coef: int, rand_range: int,
                 model: BaseModelSARSA, trainer: BaseTrainerSARSA):
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.rand_coef = rand_coef
        self.rand_range = rand_range
        self.model = model
        self.trainer = trainer

        self.n_episode = 0
        self.epsilon = 0
        self.memory: Deque[MemoryCell] = deque(maxlen=self.max_memory)

        self.model.to(self.trainer.device)

    def remember(self, memory_cell: MemoryCell):
        self.memory.append(memory_cell)

    def train_short_memory(self, memory_cell: MemoryCell):
        state = [memory_cell.state]
        action = [memory_cell.action]
        reward = [[memory_cell.reward]]
        next_state = [memory_cell.next_state] if memory_cell.next_state is not None else [[]]
        next_action = [memory_cell.next_action] if memory_cell.next_action is not None else [[]]
        done = [memory_cell.done if memory_cell.done is not None else False]

        self.trainer.train_step(state, action, reward, next_state, next_action, done)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(list(self.memory), self.batch_size)
        else:
            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, next_actions, dones = [], [], [], [], [], []
        for cell in mini_sample:
            states.append(cell.state)
            actions.append(cell.action)
            rewards.append([cell.reward])
            next_states.append(cell.next_state if cell.next_state is not None else [])
            next_actions.append(cell.next_action if cell.next_action is not None else [])
            dones.append(cell.done if cell.done is not None else False)

        self.trainer.train_step(states, actions, rewards, next_states, next_actions, dones)

    def act(self, state: List[int]) -> List[int]:
        self.epsilon = self.rand_coef - self.n_episode
        final_move = [0] * self.model.num_classes

        if random.randint(0, self.rand_range) < self.epsilon:
            move_idx = random.randint(0, self.model.num_classes - 1)
            final_move[move_idx] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.trainer.device)
            with torch.no_grad():
                prediction = self.model(state_t)
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move


class BaseAgentA2C:
    """
    Base A2C agent class
    """
    def __init__(self, actor_model: BaseModelA2CActor, value_model: BaseModelA2CValue, trainer: BaseTrainerA2C):
        self.actor_model = actor_model
        self.value_model = value_model
        self.trainer = trainer
        self.n_episode = 0

        self.actor_model.to(self.trainer.device)
        self.value_model.to(self.trainer.device)

    def train_episode(self, memory: List[MemoryCell]):
        if not memory:
            return

        states, actions, rewards = [], [], []
        for cell in memory:
            states.append(cell.state)
            actions.append(cell.action)
            rewards.append([cell.reward])  # Trainer expects List[List[float]] for rewards

        self.trainer.train_step(states, actions, rewards)

    def act(self, state: List[int]) -> List[int]:
        final_move = [0] * self.actor_model.num_classes  # Use actor model for num_classes

        state_t = torch.tensor(state, dtype=torch.float32, device=self.trainer.device)
        with torch.no_grad():
            logits = self.actor_model(state_t).squeeze(0)  # Remove batch dim for multinomial

        probs = F.softmax(logits, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()
        final_move[move_idx] = 1

        return final_move


class BaseAgentPPO:
    """
    Base PPO agent class
    """
    def __init__(self, actor_model: BaseModelPPOActor, value_model: BaseModelPPOValue, trainer: BaseTrainerPPO):
        self.actor_model = actor_model
        self.value_model = value_model
        self.trainer = trainer
        self.n_episode = 0

        self.actor_model.to(self.trainer.device)
        self.value_model.to(self.trainer.device)

    def train_episode(self, memory: List[PPOMemoryCell]):
        if not memory:
            return

        states, actions, logits_list, log_probs_list, rewards = [], [], [], [], []
        for cell in memory:
            states.append(cell.state)
            actions.append(cell.action)
            logits_list.append(cell.logits)
            log_probs_list.append(cell.log_probs)  # This is List[float] of size 1
            rewards.append([cell.reward])

        self.trainer.train_step(states, actions, logits_list, log_probs_list, rewards)

    def act(self, state: List[int]) -> Tuple[List[int], List[float], List[float]]:
        final_move = [0] * self.actor_model.num_classes

        state_t = torch.tensor(state, dtype=torch.float32, device=self.trainer.device)

        with torch.no_grad():
            logits = self.actor_model(state_t)  # [1, num_classes]

        probs = F.softmax(logits, dim=-1)
        action_tensor = torch.multinomial(probs, 1)  # [1] (tensor containing action index)
        move_idx = action_tensor.item()
        log_probs_tensor = -F.cross_entropy(logits.unsqueeze(0), action_tensor, reduction='none')  # [1]

        final_move[move_idx] = 1

        # Convert tensors to lists of floats
        logits_v = logits.cpu().float().tolist()  # List of floats for all actions
        log_probs_v = [log_probs_tensor.cpu().float().item()]  # List containing single log_prob of chosen action

        return final_move, logits_v, log_probs_v


class BaseEnvironment:
    """
    Base class for environment implementation
    """

    def __init__(self):
        self.episode_iteration: int = 0

    def define_obstacles(self):
        """
        Method to define obstacles
        """
        raise NotImplementedError

    def place_goal(self):
        """
        Method to place a goal
        """
        raise NotImplementedError

    def reset(self):
        """
        Method to reset an episode
        """
        self.episode_iteration = 0
        # Implement actual reset logic in derived classes
        raise NotImplementedError

    def get_state(self) -> List[int]:
        """
        Method to get an environment state
        """
        raise NotImplementedError

    def move(self, action: List[int]):
        """
        Method to make agent action steps (update internal state based on action)
        """
        # This method might be more about internal state update after an action.
        # The actual consequences (reward, new_state, done) are typically part of make_action.
        raise NotImplementedError

    def set_reward(self) -> float:
        """
        Method to calculate a reward for an applied action (or current state)
        This might be called internally by make_action or step.
        """
        raise NotImplementedError

    def is_collision(self) -> bool:
        """
        Method to check collisions
        """
        raise NotImplementedError

    def reach_goal(self) -> bool:
        """
        Method to check if goal is reached
        """
        raise NotImplementedError

    def make_action(self, action: List[int]) -> ActionResult:
        """
        Method to make an action and return the result (new_state, reward, done).
        This is often called `step` in RL environments.
        """
        self.episode_iteration += 1
        # Call self.move(action) or implement movement logic here
        # Call self.set_reward() or calculate reward here
        # Call self.is_collision(), self.reach_goal() to determine `done`
        # Get new_state
        raise NotImplementedError
        # Example structure:
        # self.move(action) # Update agent's position based on action
        # new_state = self.get_state()
        # reward = self.set_reward() # Calculate reward based on new state/action
        # done = self.is_collision() or self.reach_goal() or self.episode_iteration >= MAX_STEPS
        # return ActionResult(next_state=new_state, reward=reward, done=done)
