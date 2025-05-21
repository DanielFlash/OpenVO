#include "BaseTrainerPPO.h"

BaseTrainerPPO::BaseTrainerPPO(BaseModelPPOActor* actorModel, BaseModelPPOValue* valueModel, const double klCoeff, const double vCoeff,
	torch::optim::Adam* optimizerAdam, torch::Device* device)
	: m_ActorModel{ actorModel }, m_ValueModel{ valueModel }, m_klCoeff{ klCoeff }, m_vCoeff{ vCoeff },
	m_optimizerAdam{ optimizerAdam }, m_device{ device } {
	if (*m_device == torch::kCUDA) {
		m_cudaEnabled = true;
	}
	else {
		m_cudaEnabled = false;
	}

	m_ActorModel->to(*m_device);
	m_ValueModel->to(*m_device);
}

void BaseTrainerPPO::trainStep(std::vector<std::vector<int>> state, std::vector<std::vector<int>> action, std::vector<std::vector<float>> logits,
	std::vector<std::vector<float>> logProbs, std::vector<std::vector<double>> reward) {

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

	auto optL = torch::TensorOptions().dtype(torch::kFloat32);
	n = logits.size();
	m = logits[0].size();
	torch::Tensor logitsT;
	if (n > 1) {
		logitsT = torch::zeros({ n, m }, optL);
		for (int i = 0; i < n; i++) {
			logitsT.slice(0, i, i + 1) = torch::from_blob(logits[i].data(), m, optL);
		}
	}
	else {
		logitsT = torch::from_blob(logits[0].data(), m, optL);
		logitsT = logitsT.view({ 1, -1 });
	}
	logitsT.to(*m_device);

	auto optLP = torch::TensorOptions().dtype(torch::kFloat32);
	n = logProbs.size();
	m = logProbs[0].size();
	torch::Tensor logProbsT;
	if (n > 1) {
		logProbsT = torch::zeros({ n, m }, optLP);
		for (int i = 0; i < n; i++) {
			logProbsT.slice(0, i, i + 1) = torch::from_blob(logProbs[i].data(), m, optLP);
		}
	}
	else {
		logProbsT = torch::from_blob(logProbs[0].data(), m, optLP);
		logProbsT = logProbsT.view({ 1, -1 });
	}
	logProbsT.to(*m_device);

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

	m_optimizerAdam->zero_grad();
	auto values_new = m_ValueModel->forward(stateT).to(*m_device);
	auto logits_new = m_ActorModel->forward(stateT).to(*m_device);
	auto advantages = rewardT - values_new;

	auto crossEntropyOpts = torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone);
	auto logProbsT_new = -torch::nn::functional::cross_entropy(logits_new, actionT, crossEntropyOpts);
	auto prob_ratio = torch::exp(logProbsT_new - logProbsT);

	auto l0 = logitsT - torch::amax(logitsT, 1, true);
	auto l1 = logits_new - torch::amax(logits_new, 1, true);
	auto e0 = torch::exp(l0);
	auto e1 = torch::exp(l1);
	auto e_sum0 = torch::sum(e0, 1, true);
	auto e_sum1 = torch::sum(e1, 1, true);
	auto p0 = e0 / e_sum0;
	auto kl = torch::sum(p0 * (l0 - torch::log(e_sum0) - l1 + torch::log(e_sum1)), 1, true);

	auto mseOpts = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
	auto v_loss = torch::nn::functional::mse_loss(values_new, rewardT, mseOpts);
	auto loss = -advantages * prob_ratio + kl * m_klCoeff + v_loss * m_vCoeff;
	loss.sum().backward();
	m_optimizerAdam->step();
}