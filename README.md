# DeepEdging

The program hedges portfolios by simulating assets with the Heston model, without transaction costs. It is developed in Python/PyTorch and uses classes to generate trajectories, compute the loss, and implement RNN and FCN networks, all managed by a main file for training and prediction.

## Getting started

Meaning of the parameters inside main.py:

* N_SAMPLES = 2048        # Set number of samples
* BATCH_SIZE = 256        # Set batch size
* N_EPOCHS = 128          # Set number of Epochs
* OUT_DIM = 2             # Delta size (only 2)
* TIMESTEPS = 30          # Set time steps
* HIDDEN_DIM = 17         # Set dim hidden layers
* N_H_SLAYERS = 3         # Set number of layers inside RNN subnet network  
* N_H_LAYERS = 4          # Set number of layers inside FCN network  
* _PRE_TRAINED = False    # True => pre-trained network (need existing model parameters)
* _TRAIN       = True     # True => Train else Inference
* _MODEL = 'RNN'          # Un-comment => use RNN network
* #_MODEL = 'FCN'         # Un-comment => use FCN network
* _HESTON = True          # True => use Heston dataset; False => use testing dataset
* _PLOT_ERROR = True      # True => plot error histogram

## How to use

python main.py

Need to install requirements.txt modules

## License

Licensed under either of
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.

Please do not ask for features, but send a Pull Request instead.
