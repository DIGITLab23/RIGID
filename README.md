# RIGID: Random-forest-based Interpretable Generative Inverse Design

[![DOI](https://zenodo.org/badge/728962538.svg)](https://doi.org/10.5281/zenodo.14014652)

Data and code for our paper [Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning](https://arxiv.org/abs/2401.00003).

![Alt text](/overview.png)

RIGID is an inverse design model that generates designs given (qualitative) functional responses as design targets. Compared to other inverse design methods, RIGID has the following advantages:

* Efficient on small data problems
* Fast training, not requiring extensive hyperparameter tuning
* Interpretable
* Can estimate the likelihood of target satisfaction for any design and sample designs based on likelihood, thus also a generative model
* Can use a single parameter (sampling threshold) to tune the trade-off between exploitation and exploration of the design space
* Can deal with both quantitative and qualitative design variables

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen, W. W., Sun, R., Lee, D., Portela, C. M., Chen, W. (2023). Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning. arXiv preprint arXiv:2401.00003.

    @article{chen2023generative,
	  title={Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning},
	  author={Chen, Wei W and Sun, Rachel and Lee, Doksoo and Portela, Carlos M and Chen, Wei},
	  journal={arXiv preprint arXiv:2401.00003},
	  year={2023}
	}

## Code Usage

### Creating A Virtual Environment

Go to the code directory. Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```

Switch to the created environment by running:
```
conda activate rf
```

### Example Design Problems

There are four example design cases: 

   1. Acoustic metamaterial design case (in the folder "acoustic")
   2. Optical metasurface design case (in the folder "optical")
   3. Synthetic design case 1 with a squared exponential test function (in the folder "test_sqexp")
   4. Synthetic design case 2 with a superposed sine test function (in the folder "test_supsin")

In each design case folder:

   1. Define design targets and change other configurations in `config.json`. A design target is represented as ranges (frequency and wavelength ranges for the acoustic and the optical metamaterial cases, respectively). For example, `[[3,4], [6,7]]` in the acoustic case means we want to generate metamaterial designs with bandgaps in 3-4 MHz and 6-7 MHz. `auto` means the target will be randomly generated.
   2. Run `python rf_forward.py` to train a random forest model for forward prediction.
   3. Run `python rf_inverse.py` to generate new designs given the specified design target.

The generated designs and the configuration of each experiment, including the design target, will be saved in the folder "results\_rf/exp\_...".

For synthetic cases, you can run `python validate.py` to evaluate the generated designs and create a plot of the sampling threshold versus the three metrics (selection rate, satisfaction rate, and average score) shown in the paper.

To evaluate generated designs in the acoustic or the optical case, you will need a simulation model (not included in this repository) to compute the responses (i.e., dispersion relations for the acoustic case and absorbance profiles for the optical case) of generated designs.

