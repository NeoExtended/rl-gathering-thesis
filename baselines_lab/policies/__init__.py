from stable_baselines.common.policies import register_policy

from baselines_lab.policies.cnn_policy import SimpleMazeCnnPolicy, GeneralCnnPolicy
from baselines_lab.policies.rnd_policy import RndPolicy

register_policy('RndPolicy', RndPolicy)
register_policy('SimpleMazeCnnPolicy', SimpleMazeCnnPolicy)
register_policy('GeneralCnnPolicy', GeneralCnnPolicy)
