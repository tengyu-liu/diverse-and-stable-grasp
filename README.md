# diverse-and-stable-grasp
This repository shares the code to replicate results from the paper **Synthesizing Diverse and Physically Stable Grasps with Arbitrary Hand Structures using Differentiable Force Closure Estimation** [[link](https://arxiv.org/abs/2104.09194)]

We tested our code with `Python 3.8`, `PyTorch 1.9` and `CUDA 11.1`. However, the code should work with any recent version of PyTorch. 

## Prerequisites
* Numpy
* Trimesh
* Plotly
* PyTorch
* [manopth](https://github.com/hassony2/manopth)

## Download data
* Signup and download the license-protected hand model file `MANO_RIGHT.pkl` from [http://mano.is.tue.mpg.de] and place it in `data/mano/`. 
* Download DeepSDF model weights and other related files from [Google Drive]() and extract into `data/`

## Run
Run `python synthesis.py` to run our grasp synthesis algorithm with 1024 parallel syntheses, a MANO hand, and spheres with random radius. Synthesized examples that satisfy the constraints in Eq. 11 are stored in `synthesis/`. The demo code `synthesis.py` supports the following arguments: 
* `--batch_size`: number of parallel syntheses. Default: `1024`
* `--n_contact`: number of contact points. Default: `5`
* `--max_physics`: number of optimization steps. Default: `10000`
* `--max_refine`: number of refinement steps. Set to `0` to turn off refinement. Default: `1000`
* `--hand_model`: choice of `['mano', 'mano_fingertip']`. Default: `'mano'`
* `--obj_model`: choice of `['bottle', 'sphere']`. Default: `'bottle'`
* `--langevin_probability`: chance of choosing Langevin dynamics over contact point sampling in optimization steps. Default: `0.85`
* `--hprior_weight`: weight of $E_\mathrm{prior}$. Default: `1`
* `--noise_size`: size of noise used in Langevin dynamics. Default: `0.1`
* `--mano_path`: path to MANO parameters. Default: `'data/mano'`
* `--output_dir`: path to store synthesis results. Default: `'synthesis'`