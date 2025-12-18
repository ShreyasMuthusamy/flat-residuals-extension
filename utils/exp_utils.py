import numpy as np
import torch

from controller.dynamics import PlanarQuadDynamicsWithDrag, PlanarQuadDynamics
import utils.sim_utils as sim_utils
import learning.train as train
from learning.models import FullResidualModel, DeepFullResidualModel, FlatResidualModel
import controller.nmpc as nmpc
import utils.eval_utils as eval_utils
from utils.eval_utils import generate_ellipse_reference, generate_lemniscate_reference
import controller.flatness as flatness
import utils.control_utils as control_utils

import scipy.linalg as spl


def train_models(exp_seed, quad_params, data_params, training_params):
    _ = torch.manual_seed(exp_seed)

    # Define the dynamics of the system
    true_dynamics = PlanarQuadDynamicsWithDrag(**quad_params)
    nominal_dynamics = PlanarQuadDynamics(**quad_params)

    # Open-loop data collection: Sample trajectories all over the workspace
    x0_range = (data_params['x_min'], -data_params['x_min'])
    x_samples, u_samples = sim_utils.sample_random_trajectories(
        true_dynamics.dot_fn,
        data_params['num_samples'],
        data_params['dt'],
        data_params['Tf'],
        x0_range,
        data_params['u_range'],
        seed=exp_seed
    )

    # Train a neural network model
    residual_model = FullResidualModel(hidden_dims=training_params['hidden_dims'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    residual_model = train.train_finite_difference(
        x_samples, u_samples, data_params['dt'], nominal_dynamics.dot_fn, residual_model,
        lr=training_params['learning_rate'],
        epochs=training_params['num_epochs'],
        batch_size=training_params['batch_size'],
        device=device
    )
    torch.save(residual_model.state_dict(), f'models/residual_model_{exp_seed}.pth')


def load_model(model_path, training_params):
    hidden_dims = training_params['hidden_dims']
    residual_model = FullResidualModel(hidden_dims=hidden_dims)
    residual_model.load_state_dict(torch.load(model_path))
    return residual_model


def eval_open_loop(name, residual_model, quad_params, sim_params):
    true_dynamics = PlanarQuadDynamicsWithDrag(**quad_params)
    ctrl = flatness.FlatQuadrotorController(
        residual_model, quad_params, torch.zeros((1, 10)), None, None, sim_params['dt']
    )

    def fcl_ellipse(x, t):
        zt = eval_utils._zt_ellipse(
            torch.tensor(t),
            sim_params['ellipse_axes'][0],
            sim_params['ellipse_axes'][1],
            sim_params['ref_ang_vel']
        )
        ut = torch.tensor(ctrl.z2u_model(zt[None, :8], zt[None, 8:]))
        return true_dynamics.dot_fn(x, ut)[0]

    def fcl_lem(x, t):
        zt = eval_utils._zt_lemniscate(
            torch.tensor(t),
            sim_params['lemniscate_axes'][0],
            sim_params['lemniscate_axes'][1],
            sim_params['ref_ang_vel']
        )
        ut = torch.tensor(ctrl.z2u_model(zt[None, :8], zt[None, 8:]))
        return true_dynamics.dot_fn(x, ut)[0]

    # Actual reference trajectories
    ellipse_traj = generate_ellipse_reference(
        sim_params['ellipse_axes'],
        sim_params['dt'],
        sim_params['Tref'],
        sim_params['ref_ang_vel']
    )

    lemniscate_traj = generate_lemniscate_reference(
        sim_params['lemniscate_axes'],
        sim_params['dt'],
        sim_params['Tref'],
        sim_params['ref_ang_vel']
    )

    for ref_type, fcl, ref in [
        ('ellipse', fcl_ellipse, ellipse_traj), ('lem', fcl_lem, lemniscate_traj)
    ]:
        x0 = ctrl.z2x_model(*torch.split(ref[:1], 8, dim=-1))
        x_traj = sim_utils.simulate_time_varying(
            fcl,
            sim_params['dt'],
            int(sim_params['Tref']/sim_params['dt']),
            torch.tensor(x0)
        )
        # Print the tracking error
        tracking_error = torch.norm(ref[:, :2] - x_traj[:-1, 0, :2], dim=-1).mean()
        print(f"Tracking error for {ref_type} trajectory with {name}: {tracking_error}")

        # Save the trajectories
        x_traj = x_traj.squeeze().detach().numpy()
        np.save(
            f'results/open-loop/x_traj_{ref_type}_{name}.npy',
            x_traj
        )


def eval_closed_loop(seed, model_path, quad_params, training_params, sim_params):
    # Define the dynamics
    true_dynamics = PlanarQuadDynamicsWithDrag(**quad_params)

    # Load the residual model
    residual_model = load_model(model_path, training_params)

    # Compare for various reference trajectories
    ellipse_traj = generate_ellipse_reference(
        sim_params['ellipse_axes'],
        sim_params['dt'],
        sim_params['Tref'],
        sim_params['ref_ang_vel']
    )
    lemniscate_traj = generate_lemniscate_reference(
        sim_params['lemniscate_axes'],
        sim_params['dt'],
        sim_params['Tref'],
        sim_params['ref_ang_vel']
    )
    for ref_type, ref in [('ellipse', ellipse_traj), ('lem', lemniscate_traj)]:
        # Construct the learned flatness controller
        AB = spl.expm(sim_params['dt'] * np.eye(5, k=1))
        A_single, B_single, C_single = AB[:4, :4], AB[:4, 4:], np.eye(4)[:3]  # Single channel integrator
        # A_single, B_single, C_single = AB[:4, :4], AB[:4, 4:], np.eye(4)[:2]  # Single channel integrator
        linear_controller = control_utils.LinearController(
            A_single, B_single, sim_params['controller_poles']
        )
        linear_observer = control_utils.LuenbergerObserver(
            A_single, B_single, C_single, sim_params['observer_poles'], x0_hat=ref[0, :8].numpy()
        )

        learned_ctrl = flatness.FlatQuadrotorController(
            residual_model,
            quad_params,
            ref.numpy(),
            linear_controller,
            linear_observer,
            sim_params['dt']
        )
        nominal_ctrl = flatness.FlatQuadrotorController(
            None, quad_params, ref.numpy(), linear_controller, linear_observer, sim_params['dt']
        )
        x0 = torch.tensor(learned_ctrl.z2x_model(*torch.split(ref[:1], 8, dim=1))[0])

        for ctrl, name in [(learned_ctrl, 'learned'), (nominal_ctrl, 'nominal')]:
            # Simulate flat controller
            torch.manual_seed(seed)
            np.random.seed(seed)
            xtraj_flat, utraj_flat, flat_time = sim_utils.simulate_closedloop(
                true_dynamics.dot_fn,
                1,
                sim_params['dt'],
                sim_params['Tref'],
                x0,
                ctrl,
                sim_params['obs_noise_std'],
                sim_params['u_noise_std'],
                seed,
                measure_time=True
            )
            tracking_error_flat = torch.norm(ref[:, :2] - xtraj_flat[0, :-1, :2], dim=-1).mean()
            control_effort = torch.norm(utraj_flat, dim=-1).mean()
            print(f"Tracking error with {name} flat controller on {ref_type} "
                  f"trajectory: {tracking_error_flat:.5f}, control effort: {control_effort:.5f}")
            np.save(f'results/closed-loop/x_traj_{ref_type}_flat_{name}_{seed}.npy', xtraj_flat.squeeze().detach().numpy())
            np.save(f'results/closed-loop/u_traj_{ref_type}_flat_{name}_{seed}.npy', utraj_flat.squeeze().detach().numpy())
            np.save(f'results/closed-loop/time_{ref_type}_flat_{name}_{seed}.npy', flat_time)

        # Simulate nonlinear MPC controller
        nmpc_ctrl = nmpc.PlanarQuadrotorMPC(
            residual_model,
            quad_params,
            sim_params['dt'],
            ref[:, :4].numpy().T
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        xtraj_mpc, utraj_mpc, mpc_time = sim_utils.simulate_closedloop(
            true_dynamics.dot_fn,
            1,
            sim_params['dt'],
            sim_params['Tref'],
            x0,
            nmpc_ctrl,
            sim_params['obs_noise_std'],
            sim_params['u_noise_std'],
            seed,
            measure_time=True
        )
        np.save(f'results/closed-loop/x_traj_{ref_type}_mpc_{seed}.npy', xtraj_mpc.squeeze().detach().numpy())
        np.save(f'results/closed-loop/u_traj_{ref_type}_mpc_{seed}.npy', utraj_mpc.squeeze().detach().numpy())
        np.save(f'results/closed-loop/time_{ref_type}_mpc_{seed}.npy', mpc_time)

        tracking_error_mpc = torch.norm(ref[:, :2] - xtraj_mpc[0, :-1, :2], dim=-1).mean()
        control_effort = torch.norm(utraj_mpc, dim=-1).mean()
        print(f"Tracking error with NMPC controller on {ref_type} "
              f"trajectory: {tracking_error_mpc:.5f}, control effort: {control_effort:.5f}")
