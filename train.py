import torch
import pufferlib.emulation
import pufferlib.vector
import pufferlib.pufferl
import numpy as np
from env import CartPoleEnv

def make_env(render_mode=None, **kwargs):
    env = CartPoleEnv(render_mode=render_mode)
    return pufferlib.emulation.GymnasiumPufferEnv(env, **kwargs)  # Pass **kwargs to handle buf or other args

# Custom Policy (simple MLP for your env)
class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.single_observation_space.shape[0]  # 4
        action_dim = env.single_action_space.n  # 2
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_shape, 128)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(128, 128)),
            torch.nn.ReLU(),
        )
        self.action_head = pufferlib.pytorch.layer_init(torch.nn.Linear(128, action_dim), std=0.01)
        self.value_head = pufferlib.pytorch.layer_init(torch.nn.Linear(128, 1), std=1.0)

    def forward(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values
    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values


# Training and evaluation
if __name__ == '__main__':
    # Define creator to handle potential kwargs like buf
    def creator(**kwargs):
        return make_env(render_mode=None, **kwargs)

    # Create vectorized envs
    vecenv = pufferlib.vector.make(
        creator,
        num_envs=8,
        backend=pufferlib.vector.Multiprocessing,  # Change to pufferlib.vector.Serial if issues persist
    )

    # Load default config and override hyperparameters using dict access
    args = pufferlib.pufferl.load_config('default')
    args['train']['total_timesteps'] = 100000
    args['train']['learning_rate'] = 0.001
    args['train']['num_steps'] = 256
    args['train']['batch_size'] = 8 * 256  # num_envs * num_steps
    args['train']['minibatch_size'] = 256  # batch_size // desired num_minibatches (e.g., 4)
    args['train']['update_epochs'] = 10
    args['train']['gamma'] = 0.99
    args['train']['gae_lambda'] = 0.95
    args['train']['clip_coef'] = 0.2
    args['train']['norm_adv'] = True
    args['train']['clip_vloss'] = True
    args['train']['ent_coef'] = 0.00
    args['train']['vf_coef'] = 0.1
    args['train']['max_grad_norm'] = 0.1
    args['train']['anneal_lr'] = True
    args['train']['seed'] = 42
    args['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['train']['torch_deterministic'] = True  # Ensure it's bool for cudnn.deterministic
    args['train']['env'] = 'CartPole'  # Added to fix KeyError in dashboard

    # Instantiate policy
    policy = Policy(vecenv.driver_env).to(args['train']['device'])

    # Logger
    logger = pufferlib.pufferl.NoLogger(args['train'])

    # Trainer (use args['train'] as config)
    trainer = pufferlib.pufferl.PuffeRL(config=args['train'], vecenv=vecenv, policy=policy, logger=logger)

    # Training loop
    for epoch in range(1000):
        trainer.evaluate()
        logs = trainer.train()

    trainer.print_dashboard()

    trainer.close()

    print("Training complete!")
