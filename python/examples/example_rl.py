import torch
import torch.optim as optim

# from RL_module import BaseAgentSARSA, BaseModelSARSA, BaseTrainerSARSA
# from data_types import MemoryCell
from Open_VO import BaseAgentSARSA, BaseModelSARSA, BaseTrainerSARSA, MemoryCell

def main():
    maxMemory = 100000
    batchSize = 1000
    randCoef = 60
    randRange = 200
    inputSize = 4
    hiddenSizes = [32, 16]
    numClasses = 4
    lr = 0.001
    gamma = 0.9
    device = torch.device('cpu')

    model = BaseModelSARSA(inputSize, hiddenSizes, numClasses)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = BaseTrainerSARSA(model, gamma, optimizer, device)
    agent = BaseAgentSARSA(maxMemory, batchSize, randCoef, randRange, model, trainer)

    oldState = [1, 0, 0, 1]
    newState = [0, 1, 0, 1]
    finalMove = agent.act(oldState)
    newMove = agent.act(newState)
    print(finalMove, newMove)

    cell = MemoryCell(
        state=oldState, action=finalMove, next_state=newState, next_action=newMove, reward=2.3, done=False
    )

    agent.train_short_memory(cell)
    agent.remember(cell)
    agent.remember(cell)
    agent.remember(cell)
    agent.train_long_memory()


if __name__ == '__main__':
    main()
