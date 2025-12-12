import numpy as np
import torch
import torch.nn as nn
import exp_utils

from dynamics import PlanarQuadDynamicsWithDrag

quad_params = {
    'm_quad': 1.0,
    'I_quad': 0.1,
    'g': 9.81,
    'F_max': 25.0,
    'tau_max': 0.5,
    'C_pd': [0.1, 0.1],  # Parasitic drag in (x, y) directions
    'C_rdx': 1e-2
}

data_params = {
    # 'num_samples': 5000,
    # 'Tf': 0.3,
    'num_samples': 5000,
    'Tf': 0.3,
    'x_min': torch.tensor([-1, -1, -0.5, -0.5, -0.05, -0.1]),
    # 'u_range': (
    #     torch.tensor((quad_params['g'] * 0.8, -quad_params['tau_max'])),
    #     torch.tensor((quad_params['F_max'], quad_params['tau_max']))
    # ),
    'u_range': (
        torch.tensor((quad_params['g'] * 0.9, -quad_params['tau_max'])),
        torch.tensor((quad_params['g'] * 1.1, quad_params['tau_max']))
    ),
    'obs_noise_std': 0.,
    'dt': 1e-2
}

training_params = {
    'hidden_dims': [32],
    'batch_size': 128,
    'num_epochs': 20,
    'learning_rate': 4e-3,
}

open_loop_sim_params = {
    'dt': 1e-2,
    'Tref': 14.0,
    'ellipse_axes': [1, 1],
    'lemniscate_axes': [1, 0.6],
    'ref_ang_vel': np.pi / 7
}

closed_loop_sim_params = {
    'dt': 1e-2,
    'Tref': 14.0,
    'ellipse_axes': [1, 1],
    'lemniscate_axes': [1, 0.6],
    'ref_ang_vel': np.pi / 7,
    'controller_poles': np.array([0.99, 0.978, 0.98, 0.991]),
    'observer_poles': np.array([0.45, 0.55, 0.4, 0.5]),
    'obs_noise_std': 1e-3,
    'u_noise_std': torch.tensor([1e-2, 1e-3]),
}

num_experiments = 30
if __name__ == '__main__':

    # # Train models
    # for seed in range(num_experiments):
    #     exp_utils.train_models(seed, quad_params, data_params, training_params)

    # ########################
    # # Open-loop evaluation #
    # ########################
    # print('-'*10, 'Open-loop evaluation', '-'*10)

    # # On nominal flat maps
    # exp_utils.eval_open_loop('nominal', None, quad_params, open_loop_sim_params)

    # # On true flat maps
    # class TrueResidualWrapper(nn.Module):
    #     def __init__(self, quad_params):
    #         super().__init__()
    #         self.true_dynamics = PlanarQuadDynamicsWithDrag(**quad_params)

    #     def forward(self, x):
    #         res = torch.cat((torch.zeros_like(x), torch.zeros_like(x[..., :2])), dim=-1)
    #         res[..., 2:4] += self.true_dynamics.drag(
    #             torch.cat((x, torch.zeros_like(x[..., :2])), dim=-1)
    #         )
    #         return res
    # exp_utils.eval_open_loop(
    #     'true', TrueResidualWrapper(quad_params), quad_params, open_loop_sim_params
    # )

    # # On learned residual models
    # for seed in range(num_experiments):
    #     model_path = f'models/residual_model_{seed}.pth'
    #     residual_model = exp_utils.load_model(model_path, training_params)
    #     exp_utils.eval_open_loop(
    #         f'learned_{seed}', residual_model, quad_params, open_loop_sim_params
    #     )

    # Closed-loop evaluation
    print('-'*10, 'Closed-loop evaluation', '-'*10)
    for seed in range(num_experiments):
        exp_utils.eval_closed_loop(
            seed,
            f'models/residual_model_{seed}.pth',
            quad_params,
            training_params,
            closed_loop_sim_params
        )
