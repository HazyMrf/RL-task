import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

class PPG:
    def __init__(
        self,
        policy,            # policy with .act(...) method
        optimizer,         # optimizer over policy.model.parameters()
        cliprange=0.2,
        value_loss_coef=0.25,
        max_grad_norm=0.5,
        # PPG-specific
        aux_loss_coef=1.0,
        N_pi=32,           # how many policy-phase updates
        N_aux=6,           # how many epochs for aux phase
        aux_mbsize=64      # minibatch size for aux phase
    ):
        """
        policy: your Policy(...) object, which has a .model (PolicyModel) that
                outputs both policy (mean,var) and value.
        trajectory["log_probs"], ["values"], etc. must be stored by the sampler.

        In standard PPG:
          - The policy updates are just standard PPO (no decoupled ratio or EWMA).
          - After some updates, we do an auxiliary phase to better train value_model.
        """
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.aux_loss_coef = aux_loss_coef
        self.N_pi = N_pi
        self.N_aux = N_aux
        self.aux_mbsize = aux_mbsize

        self.state_buffer = []
        self.value_target_buffer = []

    def policy_loss(self, trajectory, act):
        """
        Compare new log_probs to old log_probs from trajectory.
        Example trajectory fields:
          trajectory["actions"], trajectory["advantages"], trajectory["log_probs"]
        act["distribution"] is the new distribution from the current policy.
        """
        actions = torch.tensor(trajectory["actions"], dtype=torch.float32)
        advantages = torch.tensor(trajectory["advantages"], dtype=torch.float32)

        old_log_probs = torch.tensor(trajectory["log_probs"], dtype=torch.float32)
        new_log_probs = act["distribution"].log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        J = ratio * advantages
        J_clipped = clipped_ratio * advantages

        return -torch.mean(torch.min(J, J_clipped))

    def value_loss(self, trajectory, act):
        """
        Compare new values to old values from trajectory, with clipping.
        Example trajectory fields:
          trajectory["values"], trajectory["value_targets"]
        """
        V = act["values"]
        V_old = torch.tensor(trajectory["values"], dtype=torch.float32)
        V_target = torch.tensor(trajectory["value_targets"], dtype=torch.float32)

        l_simple = (V - V_target) ** 2
        V_clipped = V_old + torch.clamp(V - V_old, -self.cliprange, self.cliprange)
        l_clipped = (V_clipped - V_target) ** 2

        return torch.mean(torch.max(l_simple, l_clipped))

    def loss(self, trajectory):
        """
        Combine policy + value losses (like standard PPO).
        Also store data for the upcoming auxiliary phase.
        """
        act = self.policy.act(trajectory["observations"], training=True)

        p_loss = self.policy_loss(trajectory, act)
        v_loss = self.value_loss(trajectory, act)
        total_loss = p_loss + self.value_loss_coef * v_loss

        self.state_buffer.append(trajectory["observations"])
        self.value_target_buffer.append(trajectory["value_targets"])

        return total_loss

    def step(self, trajectory):
        """
        Typical usage: for each batch/trajectory, call step once.
        Then, after N_pi updates, proceed to an aux phase.
        """
        self.optimizer.zero_grad()
        total_loss = self.loss(trajectory)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return total_loss.item()

    def auxiliary_loss(self, states, value_targets):
        """
        We'll feed states into self.policy.model.get_value(...) and
        do MSE with the target returns.
        Because PolicyModel has separate policy_model & value_model, we can
        freeze one part and train the other.
        """
        obs_t = torch.tensor(states, dtype=torch.float32)
        vt = torch.tensor(value_targets, dtype=torch.float32)

        pred_values = self.policy.model.get_value(obs_t).squeeze(-1)
        return F.mse_loss(pred_values, vt)

    def run_aux_phase(self):
        if not self.state_buffer:
            return

        states_np = np.concatenate(self.state_buffer, axis=0)
        vtargets_np = np.concatenate(self.value_target_buffer, axis=0)
        data_size = states_np.shape[0]

        # Freeze policy network
        for p in self.policy.model.policy_model.parameters():
            p.requires_grad = False
        # Unfreeze value network
        for p in self.policy.model.value_model.parameters():
            p.requires_grad = True

        for _ in range(self.N_aux):
            perm = np.random.permutation(data_size)
            start_i = 0
            while start_i < data_size:
                end_i = start_i + self.aux_mbsize
                inds = perm[start_i:end_i]

                batch_states = states_np[inds]
                batch_returns = vtargets_np[inds]

                self.optimizer.zero_grad()
                aux_loss_val = self.auxiliary_loss(batch_states, batch_returns)
                aux_loss_val = aux_loss_val * self.aux_loss_coef
                aux_loss_val.backward()

                nn.utils.clip_grad_norm_(self.policy.model.value_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start_i = end_i

        self.state_buffer.clear()
        self.value_target_buffer.clear()

        # Unfreeze policy
        for p in self.policy.model.policy_model.parameters():
            p.requires_grad = True




class EWMA_PPG:
    def __init__(
        self,
        policy,             # instance of your Policy(...) class
        optimizer,          # torch optimizer for entire model
        cliprange=0.2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        beta_ewma=0.99,
        # PPG hyperparams
        aux_loss_coef=1.0,
        N_pi=32,            # policy-phase gradient updates
        N_aux=6,            # epochs in the auxiliary phase
        aux_mbsize=64,      # batch size for the auxiliary phase
    ):
        """
        policy: a Policy(...) with .model = PolicyModel(...)
        beta_ewma: decay for your “proximal policy” exponential moving average
        N_pi, N_aux, aux_mbsize: standard PPG structure
        """
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.policy_prox = copy.deepcopy(policy)
        for param in self.policy_prox.model.parameters():
            param.requires_grad = False
        self.beta_ewma = beta_ewma

        self.aux_loss_coef = aux_loss_coef
        self.N_pi = N_pi
        self.N_aux = N_aux
        self.aux_mbsize = aux_mbsize

        self.state_buffer = []
        self.value_target_buffer = []

    def update_prox_policy(self):
        with torch.no_grad():
            for p_main, p_prox in zip(self.policy.model.parameters(),
                                      self.policy_prox.model.parameters()):
                p_prox.data.mul_(self.beta_ewma).add_(
                    p_main.data, alpha=(1.0 - self.beta_ewma)
                )

    def policy_loss(self, trajectory, curr_act, prox_act):
        """
        trajectory: dict with "actions", "advantages", etc.
        curr_act, prox_act: output of .act(..., training=True/False) for current/prox
        """
        actions = torch.tensor(trajectory["actions"], dtype=torch.float32)
        advantages = torch.tensor(trajectory["advantages"], dtype=torch.float32)

        new_log_probs = curr_act["distribution"].log_prob(actions)
        with torch.no_grad():
            old_log_probs = prox_act["distribution"].log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        policy_loss_ = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

        return policy_loss_

    def value_loss(self, trajectory, curr_act, prox_act):
        """
        Value clipping w.r.t. 'prox' old values
        """
        returns = torch.tensor(trajectory["value_targets"], dtype=torch.float32)

        values = curr_act["values"]
        with torch.no_grad():
            old_values = prox_act["values"]

        unclipped = (values - returns) ** 2
        clipped_val = old_values + torch.clamp(values - old_values, -self.cliprange, self.cliprange)
        clipped = (clipped_val - returns) ** 2

        return torch.mean(torch.max(unclipped, clipped))

    def policy_phase_loss(self, trajectory):
        """
        Combine decoupled PPO's policy + value losses.
        Also store states+value_targets for the upcoming aux phase.
        """
        curr_act = self.policy.act(trajectory["observations"], training=True)
        with torch.no_grad():
            prox_act = self.policy_prox.act(trajectory["observations"], training=True)

        p_loss = self.policy_loss(trajectory, curr_act, prox_act)
        v_loss = self.value_loss(trajectory, curr_act, prox_act)
        total_loss = p_loss + self.value_loss_coef * v_loss

        self.state_buffer.append(trajectory["observations"])
        self.value_target_buffer.append(trajectory["value_targets"])

        return total_loss

    def step(self, trajectory):
        self.optimizer.zero_grad()
        loss_val = self.policy_phase_loss(trajectory)
        loss_val.backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update the prox policy
        self.update_prox_policy()

        return loss_val.item()

    def auxiliary_loss(self, states, value_targets):
        """
        We'll re-run the value model on stored states, ignoring the policy part.
        Because your model has separate policy_model and value_model,
        you can freeze the policy part specifically.
        """
        obs_t = torch.tensor(states, dtype=torch.float32)
        vt = torch.tensor(value_targets, dtype=torch.float32)

        # Evaluate the current model’s value
        pred_values = self.policy.model.get_value(obs_t).squeeze(-1)
        return torch.mean((pred_values - vt) ** 2)

    def run_aux_phase(self):
        # If nothing in the buffer, do nothing
        if len(self.state_buffer) == 0:
            return

        # Flatten all states/returns in the buffer
        S = np.concatenate(self.state_buffer, axis=0)
        V = np.concatenate(self.value_target_buffer, axis=0)
        data_size = S.shape[0]

        # Freeze policy parameters (the policy_model)
        for p in self.policy.model.policy_model.parameters():
            p.requires_grad = False
        # Make sure the value part is trainable
        for p in self.policy.model.value_model.parameters():
            p.requires_grad = True

        for epoch in range(self.N_aux):
            perm = np.random.permutation(data_size)
            start_i = 0
            while start_i < data_size:
                end_i = start_i + self.aux_mbsize
                inds = perm[start_i:end_i]

                batch_states = S[inds]
                batch_returns = V[inds]

                self.optimizer.zero_grad()
                aux_loss_val = self.auxiliary_loss(batch_states, batch_returns)
                # Weighted by aux_loss_coef
                aux_loss_val = aux_loss_val * self.aux_loss_coef
                aux_loss_val.backward()

                nn.utils.clip_grad_norm_(self.policy.model.value_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start_i = end_i

        self.state_buffer.clear()
        self.value_target_buffer.clear()

        # Unfreeze the policy part for the next policy phase
        for p in self.policy.model.policy_model.parameters():
            p.requires_grad = True

