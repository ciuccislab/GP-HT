# Gaussian-Process Hilbert Transform (GP-HT)

## Project

This repository contains a minimalistic description of the source code used for the paper entitled "The Gaussian Process Hilbert Transform (GP-HT): Testing the Consistency of Electrochemical Impedance Spectroscopy Data" https://iopscience.iop.org/article/10.1149/1945-7111/aba937. The preprint is also available in the docs folder.

## Description
Electrochemical impedance spectroscopy (EIS) is an experimental technique that is frequently used in electrochemistry. While measuring EIS spectra is relatively simple, the interpretation of the data is hindered if the electrochemical system under study is not linear, causal, stable, and time invariant. These requirements can be assessed by applying the Kramers-Kronig relations or Hilbert transform (HT) to the EIS data. Here, we develop a new probabilistic approach to compute the HT of impedance spectra. The method, called the Gaussian process Hilbert transform (GP-HT), leverages Gaussian processes that are constrained to comply with the HT. The GP-HT, which is an infinite-dimensional generalization of previously developed Bayesian HT, is used to predict the credible intervals of the real part of the EIS spectrum from its imaginary component. The quality of the measurements can then be assessed by benchmarking the prediction against the real part of the experimental EIS spectrum. 

![GraphModel diagram](resources/Fig_1.jpg)
<div align='center'><strong>Figure 1. Schematic illustration of the GP-HT method.</strong></div>

## Dependencies

`numpy`

`scipy`

`matplotlib`

`seaborn`

`pandas`


## Tutorials

* **ex1_ZARC_L.ipynb**: this notebook shows how one can compute the GP-HT of a L+ZARC model.

## Citation

```
@article{ciucci2020GPHT,
  author={Ciucci, Francesco},
	title={The Gaussian Process Hilbert Transform (GP-HT): Testing the Consistency of Electrochemical Impedance Spectroscopy Data},
  journal={Journal of the Electrochemical Society},
	url={http://iopscience.iop.org/10.1149/1945-7111/aba937},
	year={2020}
  }
```

## References

1. Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139 [doi.org/10.1016/j.coelec.2018.12.003](https://doi.org/10.1016/j.coelec.2018.12.003)
2. Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. [doi.org/10.1016/j.electacta.2014.09.058](https://doi.org/10.1016/j.electacta.2014.09.058)
3. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. [doi.org/10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097)
4. Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. [doi.org/10.1016/j.electacta.2015.03.123](https://doi.org/10.1016/j.electacta.2015.03.123)
5. Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. [doi.org/10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050)
6. Liu, J., & Ciucci, F. (2019). The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. [doi.org/10.1016/j.electacta.2019.135316](https://doi.org/10.1016/j.electacta.2019.135316)
7. Liu, J., & Ciucci, F. (2020). The Deep-Prior distribution of relaxation Times. Journal of The Electrochemical Society, 167(2), 026506.[10.1149/1945-7111/ab631a](https://iopscience.iop.org/article/10.1149/1945-7111/ab631a/meta)
