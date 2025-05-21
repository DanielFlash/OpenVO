#include "BaseTrainerA2C.h"

BaseTrainerA2C::BaseTrainerA2C(BaseModelA2CActor* actorModel, BaseModelA2CValue* valueModel, torch::optim::Adam* optimizerAdamActor,
	torch::optim::Adam* optimizerAdamValue, torch::Device* device)
	: m_ActorModel{ actorModel }, m_ValueModel{ valueModel }, m_optimizerAdamActor{ optimizerAdamActor },
	m_optimizerAdamValue{ optimizerAdamValue }, m_device{ device } {
	if (*m_device == torch::kCUDA) {
		m_cudaEnabled = true;
	}
	else {
		m_cudaEnabled = false;
	}

	m_ActorModel->to(*m_device);
	m_ValueModel->to(*m_device);
}

void BaseTrainerA2C::trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action,
	std::vector<std::vector<double>> reward) {

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

	auto optAct = torch::TensorOptions().dtype(torch::kFloat32);
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

	auto optRe = torch::TensorOptions().dtype(torch::kFloat32);
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

	m_optimizerAdamValue->zero_grad();
	auto values_cr = m_ValueModel->forward(stateT).to(*m_device);
	auto mseOpts = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
	auto v_loss = torch::nn::functional::mse_loss(values_cr, rewardT, mseOpts);
	v_loss.sum().backward();
	m_optimizerAdamValue->step();

	torch::Tensor values_act;
	{
		torch::NoGradGuard no_grad;
		values_act = m_ValueModel->forward(stateT).to(*m_device);
	}
	m_optimizerAdamActor->zero_grad();
	auto advantages = rewardT - values_act;
	auto logits = m_ActorModel->forward(stateT).to(*m_device);
	auto crossEntropyOpts = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
	auto log_probs = -torch::nn::functional::cross_entropy(logits, actionT, crossEntropyOpts);
	auto pi_loss = -log_probs * advantages;
	pi_loss.sum().backward();
	m_optimizerAdamActor->step();
}