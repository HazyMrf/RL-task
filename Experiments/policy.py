from torch.distributions.multivariate_normal import MultivariateNormal
import torch

from torch import nn
from torch.nn import functional as F
import torch


class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h = 64

        self.policy_model = nn.Sequential(
            nn.Linear(state_dim, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 2 * action_dim)
        )

        self.value_model = nn.Sequential(
            nn.Linear(state_dim, self.h),
            nn.Tanh(),
            nn.Linear(self.h, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 1)
        )

    def get_policy(self, x):
        result = self.policy_model(x)
        mean, var = result.chunk(2, dim=-1)
        return mean, F.softplus(var)

    def get_value(self, x):
        return self.value_model(x)

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value

class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs, training=False):
        # Free colab dont have enough cuda :(
        inputs = torch.tensor(inputs, dtype=torch.float32, device="cpu")

        mean, var = self.model.get_policy(inputs)
        cov_matrix = torch.diag_embed(var)
        normal_distr = MultivariateNormal(mean, cov_matrix)

        actions = normal_distr.sample()
        log_probs = normal_distr.log_prob(actions)

        values = self.model.get_value(inputs)

        if training:
            return {"distribution": normal_distr, "values": values.squeeze()}

        return {
            "actions": actions.cpu().numpy().squeeze(),
            "log_probs": log_probs.detach().cpu().numpy().squeeze(),
            "values": values.detach().cpu().numpy().squeeze(),
        }
    

