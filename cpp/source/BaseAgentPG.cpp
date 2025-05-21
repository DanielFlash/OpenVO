#include "BaseAgentPG.h"

BaseAgentPG::BaseAgentPG(BaseModelPG* model, BaseTrainerPG* trainer)
	: m_Model{ model }, m_Trainer{ trainer } {

	m_Model->to(*(m_Trainer->m_device));
}

void BaseAgentPG::trainEpisode(std::vector<MemoryCell> memory) {
	std::vector<std::vector<int>> state{};
	std::vector<std::vector<int>> action{};
	std::vector<std::vector<double>> reward{};

	for (MemoryCell& cell : memory) {
		state.push_back(cell.state);
		action.push_back(cell.action);
		reward.push_back(std::vector<double>{ cell.reward });
	}

	m_Trainer->trainStep(state, action, reward);
}

std::vector<int> BaseAgentPG::act(std::vector<int> state) {
	std::vector<int> finalMove{};
	for (int i = 0; i < m_Model->m_numClasses; i++) {
		finalMove.push_back(0);
	}

	auto optSt = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor stateT = torch::from_blob(state.data(), state.size(), optSt);
	stateT = stateT.view({ 1, -1 });
	stateT.to(*(m_Trainer->m_device));
	auto logits = m_Model->forward(stateT).to(*(m_Trainer->m_device));
	logits = logits.squeeze(0);
	auto softmaxOpts = torch::nn::functional::SoftmaxFuncOptions(-1);
	auto probs = torch::nn::functional::softmax(logits, softmaxOpts);
	auto move = torch::multinomial(probs, 1).item<int>();
	finalMove[move] = 1;

	return finalMove;
}