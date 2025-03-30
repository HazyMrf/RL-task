import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

class PPG:
    def __init__(self, policy, optimizer, cliprange=0.2, value_loss_coef=0.25, aux_loss_coef=0.1, max_grad_norm=0.5, aux_phase_freq=6):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.aux_loss_coef = aux_loss_coef  # Коэффициент для вспомогательной фазы
        self.max_grad_norm = max_grad_norm
        self.aux_phase_freq = aux_phase_freq  # Частота запуска auxiliary фазы
        self.step_count = 0

    def policy_loss(self, trajectory, act):
        advantages, actions, distr = torch.tensor(trajectory["advantages"]), torch.tensor(trajectory["actions"]), act["distribution"]
        old_log_probs = torch.tensor(trajectory["log_probs"])
        new_log_probs = distr.log_prob(actions)

        probs_coeff = torch.exp(new_log_probs - old_log_probs)
        J = probs_coeff * advantages
        J_clipped = torch.clamp(probs_coeff, 1 - self.cliprange, 1 + self.cliprange) * advantages
        return -torch.mean(torch.min(J, J_clipped))

    def value_loss(self, trajectory, act):
        V, V_old, V_target = act['values'], torch.tensor(trajectory['values']), torch.tensor(trajectory["value_targets"])
        l_simple = (V - V_target) ** 2
        l_clipped = (V_old + torch.clamp(V - V_old, -self.cliprange, self.cliprange) - V_target) ** 2
        return torch.mean(torch.max(l_simple, l_clipped))
    
    def auxiliary_loss(self, trajectory):
        """ Дополнительное обновление функции ценности (auxiliary phase). """
        act = self.policy.act(trajectory["observations"], training=True)
        V, V_target = act['values'], torch.tensor(trajectory["value_targets"])
        return torch.mean((V - V_target) ** 2)
    
    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss_value = self.policy_loss(trajectory, act)
        value_loss_value = self.value_loss(trajectory, act)
        return policy_loss_value + self.value_loss_coef * value_loss_value

    def step_policy_phase(self, trajectory):
        """ Шаг policy фазы: обновление политики без обновления ценности. """
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Запуск auxiliary фазы раз в aux_phase_freq шагов
        self.step_count += 1
        if self.step_count % self.aux_phase_freq == 0:
            self.step_auxiliary_phase(trajectory)
    
    def step_auxiliary_phase(self, trajectory):
        """ Шаг auxiliary фазы: обновление только функции ценности. """
        self.optimizer.zero_grad()
        aux_loss = self.auxiliary_loss(trajectory) * self.aux_loss_coef
        aux_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
