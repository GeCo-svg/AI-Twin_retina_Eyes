import torch, torch.nn as nn, torch.optim as optim
from torch.distributions.normal import Normal
class PPO:
    def __init__(self, model, lr=3e-4, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01):
        self.model=model; self.opt=optim.Adam(model.parameters(), lr=lr)
        self.clip_ratio=clip_ratio; self.vf_coef=vf_coef; self.ent_coef=ent_coef
    def policy(self, obs):
        mu,log_std,v=self.model(obs); dist=Normal(mu, log_std.exp()); return dist, v
    def update(self, batch, epochs=5, batch_size=256):
        obs=torch.as_tensor(batch["obs"],dtype=torch.float32)
        act=torch.as_tensor(batch["act"],dtype=torch.float32)
        adv=torch.as_tensor(batch["adv"],dtype=torch.float32)
        ret=torch.as_tensor(batch["ret"],dtype=torch.float32)
        logp_old=torch.as_tensor(batch["logp"],dtype=torch.float32)
        N=len(obs)
        for _ in range(epochs):
            idx=torch.randperm(N)
            for i in range(0,N,batch_size):
                j=idx[i:i+batch_size]
                o,a,ad,r,lpo=obs[j],act[j],adv[j],ret[j],logp_old[j]
                dist,v=self.policy(o); logp=dist.log_prob(a).sum(-1)
                ratio=torch.exp(logp - lpo)
                clip_adv=torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*ad
                pi_loss=-(torch.min(ratio*ad, clip_adv)).mean()
                v_loss=((v.squeeze(-1)-r)**2).mean()
                ent=dist.entropy().sum(-1).mean()
                loss=pi_loss + self.vf_coef*v_loss - self.ent_coef*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.opt.step()
