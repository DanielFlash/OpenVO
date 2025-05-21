#include "BaseAgentDQN.h"

BaseAgentDQN::BaseAgentDQN(const int maxMemory, const int batchSize, const int randCoef, const int randRange, BaseModelDQN* model, BaseTrainerDQN* trainer)
	: m_maxMemory{ maxMemory }, m_batchSize{ batchSize }, m_randCoef{ randCoef }, m_randRange{ randRange }, m_Model{ model }, m_Trainer{ trainer } {

	m_Model->to(*(m_Trainer->m_device));
}

void BaseAgentDQN::remember(MemoryCell memoryCell) {
	if (m_memory.size() == m_maxMemory) {
		m_memory.pop_front();
	}

	m_memory.push_back(memoryCell);
}

void BaseAgentDQN::trainShortMemory(MemoryCell memoryCell) {
	std::vector<std::vector<int>> state{ memoryCell.state };
	std::vector<std::vector<int>> action{ memoryCell.action };
	std::vector<std::vector<int>> nextState{ memoryCell.nextState };
	std::vector<std::vector<double>> reward{ std::vector<double>{ memoryCell.reward } };
	std::vector<bool> done{ memoryCell.done };

	m_Trainer->trainStep(state, action, reward, nextState, done);
}

void BaseAgentDQN::trainLongMemory() {
	std::deque<MemoryCell> memory{};
	if (m_memory.size() > m_batchSize) {
		std::sample(m_memory.begin(), m_memory.end(), std::back_inserter(memory), m_batchSize, std::mt19937{ std::random_device{}() });
	}
	else {
		memory = m_memory;
	}

	std::vector<std::vector<int>> state{};
	std::vector<std::vector<int>> action{};
	std::vector<std::vector<int>> nextState{};
	std::vector<std::vector<double>> reward{};
	std::vector<bool> done{};

	for (MemoryCell& cell : memory) {
		state.push_back(cell.state);
		action.push_back(cell.action);
		nextState.push_back(cell.nextState);
		reward.push_back(std::vector<double>{ cell.reward });
		done.push_back(cell.done);
	}

	m_Trainer->trainStep(state, action, reward, nextState, done);
}

std::vector<int> BaseAgentDQN::act(std::vector<int> state) {
	m_epsilon = m_randCoef - m_nEpisode;
	std::vector<int> finalMove{};
	for (int i = 0; i < m_Model->m_numClasses; i++) {
		finalMove.push_back(0);
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib1(0, m_randRange);
	std::uniform_int_distribution<> distrib2(0, m_Model->m_numClasses);
	if (distrib1(gen) < m_epsilon) {
		int move = distrib2(gen);
		finalMove[move] = 1;
	}
	else {
		auto optSt = torch::TensorOptions().dtype(torch::kFloat32);
		torch::Tensor stateT = torch::from_blob(state.data(), state.size(), optSt);
		stateT = stateT.view({ 1, -1 });
		stateT.to(*(m_Trainer->m_device));
		auto pred = m_Model->forward(stateT).to(*(m_Trainer->m_device));
		auto move = torch::argmax(pred).item<int>();
		finalMove[move] = 1;
	}

	return finalMove;
}