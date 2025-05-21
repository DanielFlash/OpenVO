#include "BaseAgentPPO.h"

BaseAgentPPO::BaseAgentPPO(BaseModelPPOActor* actorModel, BaseModelPPOValue* valueModel, BaseTrainerPPO* trainer)
	: m_ActorModel{ actorModel }, m_ValueModel{ valueModel }, m_Trainer{ trainer } {

	m_ActorModel->to(*(m_Trainer->m_device));
	m_ValueModel->to(*(m_Trainer->m_device));
}

void BaseAgentPPO::trainEpisode(std::vector<PPOMemoryCell> memory) {
	std::vector<std::vector<int>> state{};
	std::vector<std::vector<int>> action{};
	std::vector<std::vector<float>> logits{};
	std::vector<std::vector<float>> logProbs{};
	std::vector<std::vector<double>> reward{};

	for (PPOMemoryCell& cell : memory) {
		state.push_back(cell.state);
		action.push_back(cell.action);
		logits.push_back(cell.logits);
		logProbs.push_back(cell.logProbs);
		reward.push_back(std::vector<double>{ cell.reward });
	}

	m_Trainer->trainStep(state, action, logits, logProbs, reward);
}

auto BaseAgentPPO::act(std::vector<int> state) -> std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> {
	std::vector<int> finalMove{};
	for (int i = 0; i < m_ActorModel->m_numClasses; i++) {
		finalMove.push_back(0);
	}

	auto optSt = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor stateT = torch::from_blob(state.data(), state.size(), optSt);
	stateT = stateT.view({ 1, -1 });
	stateT.to(*(m_Trainer->m_device));
	auto logits = m_ActorModel->forward(stateT).to(*(m_Trainer->m_device));
	logits = logits.squeeze(0);
	auto softmaxOpts = torch::nn::functional::SoftmaxFuncOptions(-1);
	auto probs = torch::nn::functional::softmax(logits, softmaxOpts);
	auto action = torch::multinomial(probs, 1);
	auto move = action.item<int>();
	action = action.squeeze(0);
	auto crossEntropyOpts = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
	auto log_probs = -torch::nn::functional::cross_entropy(logits, action, crossEntropyOpts);

	finalMove[move] = 1;
	logits.to(torch::kCPU).toType(torch::kFloat32);
	std::vector<float> logitsV(logits.data_ptr<float>(), logits.data_ptr<float>() + logits.numel());
	log_probs.to(torch::kCPU).toType(torch::kFloat32);
	std::vector<float> logProbsV(log_probs.data_ptr<float>(), log_probs.data_ptr<float>() + log_probs.numel());

	return std::make_tuple(finalMove, logitsV, logProbsV);
}