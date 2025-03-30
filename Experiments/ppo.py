import numpy as np
import torch
import torch.nn as nn
import copy

from utils import AsArray
from env import EnvRunner
class GAE:
    """Generalized Advantage Estimator."""

    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory):
        gamma = self.gamma
        lambda_ = self.lambda_

        # GAE results
        trajectory['advantages'] = []
        trajectory['value_targets'] = []

        r = trajectory['rewards']
        resets = trajectory['resets'] # check terminal state

        V_pi = list(trajectory["values"])
        latest_observation_act_results = self.policy.act(trajectory['state']['latest_observation'])
        V_pi.append(latest_observation_act_results['values'])

        # Compute advantages in reverse -> no need to recalc the sum on each iter
        advantage = 0.0
        for l in reversed(range(len(V_pi) - 1)):
            delta_t = r[l] + gamma * V_pi[l + 1] * ~resets[l] - V_pi[l]

            # Recalc advantage for next t
            advantage = advantage * gamma * lambda_ * ~resets[l]
            advantage += delta_t

            value_target = advantage + V_pi[l]

            trajectory['advantages'].append(advantage)
            trajectory['value_targets'].append(value_target)

        # Reverse calculated advantages
        trajectory['advantages'] = np.array(trajectory['advantages'][::-1])
        trajectory['value_targets'] = np.array(trajectory['value_targets'][::-1])


def flatten_first_two_dims(arr):
    if arr.ndim == 2:
        return arr.reshape(-1)
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


class TrajectorySampler:
    """Samples minibatches from trajectory for a number of epochs."""

    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None

    def shuffle_trajectory(self):
        """Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        trajectory_len = len(self.trajectory["observations"])

        permutation = np.random.permutation(trajectory_len)
        for key, value in self.trajectory.items():
            if key != "state":
                self.trajectory[key] = value[permutation]

    def squeeze_trajectory(self):
        for key, value in self.trajectory.items():
            if key != "state":
                self.trajectory[key] = flatten_first_two_dims(value)

    def get_trajectory(self):
        self.trajectory = self.runner.get_next()
        self.squeeze_trajectory()

    def get_next(self):
        """Returns next minibatch."""
        if not self.trajectory:
            self.get_trajectory()

        if self.minibatch_count == self.num_minibatches:
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.get_trajectory()
            self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len // self.num_minibatches

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != "state":
                minibatch[key] = value[
                    self.minibatch_count
                    * batch_size : (self.minibatch_count + 1)
                    * batch_size
                ]

        self.minibatch_count += 1

        for transform in self.transforms:
            transform(minibatch)

        return minibatch
    
class NormalizeAdvantages:
    """Normalizes advantages to have zero mean and unit std."""

    def __call__(self, trajectory):
        advantages = trajectory["advantages"]

        mean = advantages.mean()
        std = advantages.std()

        trajectory["advantages"] = (advantages - mean) / std


class PPO:
    def __init__(
        self, policy, optimizer, cliprange=0.2, value_loss_coef=0.25, max_grad_norm=0.5
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """Computes and returns policy loss on a given trajectory."""
        advantages, actions, distr = torch.tensor(trajectory["advantages"]), torch.tensor(trajectory["actions"]), act["distribution"]

        old_log_probs = torch.tensor(trajectory["log_probs"])
        new_log_probs = distr.log_prob(actions)

        probs_coeff = torch.exp(new_log_probs - old_log_probs)

        J = probs_coeff * advantages
        J_clipped = torch.clamp(probs_coeff, 1 - self.cliprange, 1 + self.cliprange) * advantages

        return -torch.mean(torch.min(J, J_clipped))

    def value_loss(self, trajectory, act):
        """Computes and returns value loss on a given trajectory."""
        V, V_old, V_target = act['values'], torch.tensor(trajectory['values']), torch.tensor(trajectory["value_targets"])

        l_simple = (V - V_target) ** 2
        l_clipped = (V_old + torch.clamp(V - V_old, -self.cliprange, self.cliprange) - V_target) ** 2

        return torch.mean(torch.max(l_simple, l_clipped))

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)

        policy_loss_value = self.policy_loss(trajectory, act)
        value_loss_value = self.value_loss(trajectory, act)

        total_loss = policy_loss_value + self.value_loss_coef * value_loss_value
        return total_loss

    def step(self, trajectory):
        """Computes the loss function and performs a single gradient step."""
        self.optimizer.zero_grad()

        loss = self.loss(trajectory).backward()
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)

        self.optimizer.step()






class DecoupledPPO:
    def __init__(self, policy, optimizer, cliprange=0.2, value_loss_coef=0.5,
                 max_grad_norm=0.5, beta_ewma=0.99):
        """
        policy: your main (live) policy model (nn.Module).
        lr, cliprange, etc.: typical PPO hyperparams.
        beta_ewma: decay rate for the EWMA “proximal policy”.
        """
        self.policy = policy  # pi_behav = pi_theta
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer

        self.policy_prox = copy.deepcopy(policy) # pi_proxy
        for p in self.policy_prox.model.parameters():
            p.requires_grad = False

        self.beta_ewma = beta_ewma

    def update_prox_policy(self):
        with torch.no_grad():
            for p_main, p_prox in zip(self.policy.model.parameters(),
                                      self.policy_prox.model.parameters()):
                p_prox.data.mul_(self.beta_ewma).add_(
                    p_main.data, alpha=(1.0 - self.beta_ewma)
                )

    def policy_loss(self, trajectory, curr_act, proxy_act):
        actions = torch.tensor(trajectory["actions"])
        advantages = torch.tensor(trajectory["advantages"], dtype=torch.float)

        current_distr = curr_act["distribution"]
        new_log_probs = current_distr.log_prob(actions)

        with torch.no_grad():
            prox_distr = proxy_act["distribution"]
            old_log_probs = prox_distr.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()

        # PPO clipping
        ratio_clipped = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        policy_loss_ = -torch.mean(torch.min(ratio * advantages, ratio_clipped * advantages))

        return policy_loss_

    def value_loss(self, trajectory, curr_act, proxy_act):
        """
        Value loss with clipping (optional).
        We'll treat self.policy as having a .value(obs) that returns V(s).
        """
        returns = torch.tensor(trajectory["value_targets"], dtype=torch.float)

        values = curr_act['values']
        with torch.no_grad():
            old_values = proxy_act['values']

        unclipped_loss = (values - returns)**2
        clipped_values = old_values + torch.clamp(values - old_values,
                                                 -self.cliprange, self.cliprange)
        clipped_loss = (clipped_values - returns)**2

        value_loss_ = torch.mean(torch.max(unclipped_loss, clipped_loss))
        return value_loss_

    def loss(self, trajectory):
        """
        Combine policy + value losses into total PPO loss.
        """
        curr_act = self.policy.act(trajectory["observations"], training=True)
        with torch.no_grad():
            prox_act = self.policy_prox.act(trajectory["observations"], training=True)

        p_loss = self.policy_loss(trajectory, curr_act, prox_act)
        v_loss = self.value_loss(trajectory, curr_act, prox_act)
        total = p_loss + self.value_loss_coef * v_loss
        return total

    def step(self, trajectory):
        """
        A single PPO update using the decoupled objective.
        Then update self.policy_prox with an EWMA.
        """
        self.optimizer.zero_grad()
        total_loss = self.loss(trajectory)
        total_loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Now update the proximal policy (EWMA)
        self.update_prox_policy()

        return total_loss.item()


def make_ppo_sampler(
    env,
    policy,
    num_runner_steps=2048,
    gamma=0.99,
    lambda_=0.95,
    num_epochs=10,
    num_minibatches=32,
):
    """Creates runner for PPO algorithm."""
    runner_transforms = [AsArray(), GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps, transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(
        runner,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        transforms=sampler_transforms,
    )
    return sampler