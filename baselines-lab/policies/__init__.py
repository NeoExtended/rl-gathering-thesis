from stable_baselines.common.policies import register_policy
from policies.rnd_policy import RndPolicy

register_policy('RndPolicy', RndPolicy)