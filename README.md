# flat-residual-extension

Final project for ESE 5460: Principles of Deep Learning. Extension of Fengjun Wang's work on learning flatness-preserving residuals for pure feedback systems.

## Running this project

To run this project on your own, you can run the script with the command `python main.py [--train] [--eval_open_loop] [--eval_closed_loop] [--num_models <number>]` on your machine. This project should train fairly quickly, even on a CPU. The script must be ran with the train option before the eval options can be used.

If you would like to just see the performance of the learned residual dynamics, you can run with 1 model (which is the default). For a more rigorous analysis, the script should be run with 10+ models. When plotting using the notebook, make sure the change the value of the `num_models` variable in the fifth cell.
