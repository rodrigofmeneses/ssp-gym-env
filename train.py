#%%
# Criando pastas para salvar os checkpoints e os resultados
import os
import shutil

# Pasta do checkpoint
chkpt_root = "tmp/exa"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
# Pasta dos resultados
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# Inicialiando o rllib e selecionando o nosso ambiente personalizado
import ray
# Inicializando o Ray
ray.init(ignore_reinit_error=True)
# Importante ambiente e registrador
from ray.tune.registry import register_env
from gym_ssp.envs.ssp_env import SSP

select_env = "ssp-v0"
register_env(select_env, lambda config: SSP())

# Importanto rllib e o algoritmo de policy proximal optimization
import ray.rllib.agents.ppo as ppo
# Parametros do ppo
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
# Nosso agente
agent = ppo.PPOTrainer(config, env=select_env)
# Status para visualização do código
status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"

#%%
n_inter = 20

for n in range(n_inter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)
    print(status.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        chkpt_file
    ))

# Agora rollouts (lançamentos)

#%%
import gym
# Importando e carregando as informação do checkpoint
agent.restore(chkpt_file)
env = gym.make(select_env)
state = env.reset()

sum_reward = 0
n_step = 30

#%%
for step in range(n_step):
    action = agent.compute_action(state)
    state, reward, done, info = env.step(action)

    sum_reward += reward
    
    if done:
        print("Cumulative reward ", sum_reward)
        state = env.reset()
        sum_reward = 0



# %%
