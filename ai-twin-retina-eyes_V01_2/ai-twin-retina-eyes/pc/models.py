import torch, torch.nn as nn
class MLPPolicyValue(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden=[256,256,256,232]; layers=[]; last=obs_dim
        for h in hidden: layers += [nn.Linear(last,h), nn.Tanh()]; last=h
        self.backbone=nn.Sequential(*layers)
        self.mu_head=nn.Linear(last, act_dim)
        self.log_std=nn.Parameter(torch.zeros(act_dim))
        self.v_head=nn.Linear(last,1)
    def forward(self, obs):
        z=self.backbone(obs); mu=self.mu_head(z); v=self.v_head(z)
        log_std=self.log_std.expand_as(mu); return mu, log_std, v
