#include "BaseTrainerSARSA.h"

BaseTrainerSARSA::BaseTrainerSARSA(BaseModelSARSA* model, const double gamma, torch::optim::Adam* optimizerAdam, torch::Device* device)
	: m_Model{ model }, m_gamma{ gamma }, m_optimizerAdam{ optimizerAdam }, m_device{ device } {
	if (*m_device == torch::kCUDA) {
		m_cudaEnabled = true;
	}
	else {
		m_cudaEnabled = false;
	}

	m_Model->to(*m_device);
}

void BaseTrainerSARSA::trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action,
	std::vector<std::vector<double>> reward, std::vector<std::vector<int>> nextState,
	std::vector<std::vector<int>> nextAction, std::vector<bool> done) {

	auto optSt = torch::TensorOptions().dtype(torch::kFloat32);
	int n = state.size();
	int m = state[0].size();
	torch::Tensor stateT;
	if (n > 1) {
		stateT = torch::zeros({ n, m }, optSt);
		for (int i = 0; i < n; i++) {
			stateT.slice(0, i, i + 1) = torch::from_blob(state[i].data(), m, optSt);
		}
	}
	else {
		stateT = torch::from_blob(state[0].data(), m, optSt);
		stateT = stateT.view({ 1, -1 });
	}
	stateT.to(*m_device);

	auto optNSt = torch::TensorOptions().dtype(torch::kFloat32);
	n = nextState.size();
	m = nextState[0].size();
	torch::Tensor nextStateT;
	if (n > 1) {
		nextStateT = torch::zeros({ n, m }, optNSt);
		for (int i = 0; i < n; i++) {
			nextStateT.slice(0, i, i + 1) = torch::from_blob(nextState[i].data(), m, optNSt);
		}
	}
	else {
		nextStateT = torch::from_blob(nextState[0].data(), m, optNSt);
		nextStateT = nextStateT.view({ 1, -1 });
	}
	nextStateT.to(*m_device);

	auto optAct = torch::TensorOptions().dtype(torch::kInt32);
	n = action.size();
	m = action[0].size();
	torch::Tensor actionT;
	if (n > 1) {
		actionT = torch::zeros({ n, m }, optAct);
		for (int i = 0; i < n; i++) {
			actionT.slice(0, i, i + 1) = torch::from_blob(action[i].data(), m, optAct);
		}
	}
	else {
		actionT = torch::from_blob(action[0].data(), m, optAct);
		actionT = actionT.view({ 1, -1 });
	}
	actionT.to(*m_device);

	auto optRe = torch::TensorOptions().dtype(torch::kDouble);
	n = reward.size();
	m = reward[0].size();
	torch::Tensor rewardT;
	if (n > 1) {
		rewardT = torch::zeros({ n, m }, optRe);
		for (int i = 0; i < n; i++) {
			rewardT.slice(0, i, i + 1) = torch::from_blob(reward[i].data(), m, optRe);
		}
	}
	else {
		rewardT = torch::from_blob(reward[0].data(), m, optRe);
		rewardT = rewardT.view({ 1, -1 });
	}
	rewardT.to(*m_device);

	auto optNAct = torch::TensorOptions().dtype(torch::kInt32);
	n = nextAction.size();
	m = nextAction[0].size();
	torch::Tensor nextActionT;
	if (n > 1) {
		nextActionT = torch::zeros({ n, m }, optNAct);
		for (int i = 0; i < n; i++) {
			nextActionT.slice(0, i, i + 1) = torch::from_blob(nextAction[i].data(), m, optNAct);
		}
	}
	else {
		nextActionT = torch::from_blob(nextAction[0].data(), m, optNAct);
		nextActionT = nextActionT.view({ 1, -1 });
	}
	nextActionT.to(*m_device);

	auto pred = m_Model->forward(stateT).to(*m_device);
	auto target = pred.clone().to(*m_device);

	for (int i = 0; i < done.size(); i++) {
		auto QNew = rewardT[i][0];
		if (!done[i]) {
			auto QNew = rewardT[i][0] + m_gamma * m_Model->forward(nextStateT[i])[torch::argmax(nextActionT).item()];
		}
		target[i][torch::argmax(actionT).item()] = QNew;
	}

	m_optimizerAdam->zero_grad();
	auto loss = torch::nn::functional::mse_loss(target, pred);
	loss.backward();
	m_optimizerAdam->step();
}