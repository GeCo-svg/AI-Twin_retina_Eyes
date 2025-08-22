import numpy as np, torch
from .models import MLPPolicyValue
from .ppo import PPO
class StubEnv:
    def __init__(self, obs_dim=24, act_dim=2): self.obs_dim=obs_dim; self.act_dim=act_dim; self.t=0
    def reset(self): self.t=0; return np.zeros(self.obs_dim, dtype=np.float32)
    def step(self, a):
        self.t+=1; obs=np.random.randn(self.obs_dim).astype(np.float32)*0.01
        r=1.0 - 0.1*np.linalg.norm(a); done=self.t>=128; return obs,r,done,{}
def rollout(env, policy, steps=1024, gamma=0.99, lam=0.95):
    obs=env.reset(); buf={"obs":[], "act":[], "ret":[], "adv":[], "logp":[]}
    vals=[]; rews=[]; lps=[]
    for t in range(steps):
        o=torch.as_tensor(obs[None],dtype=torch.float32); dist,v=policy(o)
        a=dist.sample(); lp=dist.log_prob(a).sum(-1).item(); a_np=a.numpy().squeeze(0)
        obs2,r,done,_=env.step(a_np)
        buf["obs"].append(obs); buf["act"].append(a_np); vals.append(v.item()); rews.append(r); lps.append(lp)
        obs=env.reset() if done else obs2
    adv=[]; gae=0.0; next_v=0.0
    for t in reversed(range(steps)):
        delta=rews[t]+gamma*next_v - vals[t]; gae=delta + gamma*lam*gae; adv.insert(0, gae); next_v=vals[t]
    ret=[a+v for a,v in zip(adv, vals)]
    buf["ret"]=ret; buf["adv"]=adv; buf["logp"]=lps
    return {k:np.array(v, dtype=np.float32) for k,v in buf.items()}
def main():
    obs_dim=24; act_dim=2
    model=MLPPolicyValue(obs_dim, act_dim); ppo=PPO(model); env=StubEnv(obs_dim, act_dim)
    for it in range(3):
        batch=rollout(env, ppo.policy, steps=1024)
        adv=batch["adv"]; batch["adv"]=(adv-adv.mean())/(adv.std()+1e-8)
        ppo.update(batch); print(f"[iter {it}] adv_mean={adv.mean():.3f}")
if __name__=="__main__": main()
