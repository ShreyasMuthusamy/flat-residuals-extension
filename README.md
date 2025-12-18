# flat-residual-extension

Final project for ESE 5460: Principles of Deep Learning. Extension of Fengjun Wang's work on learning flatness-preserving residuals for pure feedback systems.

## Running this project

To run this project on your own, you can run the script with the command `python main.py --parametrization {flat,fcnn} --disturbance {1,2,3} [--train] [--eval_open_loop] [--eval_closed_loop] [--num_models NUM_MODELS]` on your machine. This project should train fairly quickly, even on a CPU. The script must be ran with the `train` option before the `eval` options can be used.

There are two required options. The `parametrization` option determines which parametrization you would like to run. If you choose `flat`, then the lower triangular parametrization is trained/evaluated. If you choose `fcnn`, then a naïve fully connected neural network is used for training/evaluation. The `disturbance` option determines which of the three true residuals you would like to test. Disturbance 1 corresponds to the lower triangular disturbance tested in Experiment 5. Disturbances 2 and 3 correspond to the non-lower triangular disturbances tested in Experiments 1/2 and 3/4, respectively.

If you would like to just see the performance of the learned residual dynamics, you can run with 1 model (which is the default). For a more rigorous analysis, the script should be run with 10+ models. When plotting using the notebook, make sure the change the value of the `num_models` variable in the fifth cell.
