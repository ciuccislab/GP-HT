# Gaussian-Process Hilbert Transform (GP-HT)

## Project

This repository contains an implementation of the source code used for the paper entitled "The Gaussian Process Hilbert Transform (GP-HT): Testing the Consistency of Electrochemical Impedance Spectroscopy Data" https://iopscience.iop.org/article/10.1149/1945-7111/aba937. The preprint is also available in the docs folder.

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
* **ex2_2xZARCs.ipynb**: this notebook shows how one can compute the GP-HT of a ZARC+ZARC model.
* **ex3_drift.ipynb**: this notebook shows an application GP-HT method for a case where an EIS spectrum does not comply with the HT because of drift. The drift frequencies are also detected.
* **ex4_MoS2.ipynb**: this notebook shows how the GP-HT can be applied to unbounded EIS data. 
* **ex5_Rct_ZARC.ipynb**: this notebook shows how one can compute the charge-transfer resistance of a ZARC model.
* **ex6_Rct_Battery.ipynb**: this notebook shows how one can compute the charge-transfer resistance for real EIS data from a lithium-metal battery.
* **ex7_ZARC_L_ALC_det.ipynb**: this notebook shows how the active learning Cohn can be applied to optimally select the frequencies for a L+ZARC model.
* **ex8_ZARC_L_ALM_det.ipynb**: this notebook shows how the active learning MacKay can be applied to optimally select the frequencies for a L+ZARC model.

## Citation

```
@article{Ciucci_2020,
	doi = {10.1149/1945-7111/aba937},
	url = {https://doi.org/10.1149/1945-7111/aba937},
	year = 2020,
	month = {aug},
	publisher = {The Electrochemical Society},
	volume = {167},
	number = {12},
	pages = {126503},
	author = {Francesco Ciucci},
	title = {The Gaussian Process Hilbert Transform ({GP}-{HT}): Testing the Consistency of Electrochemical Impedance Spectroscopy Data},
	journal = {Journal of The Electrochemical Society}
}
```
```
@article{Py_2023,
	doi = {10.1016/j.electacta.2022.141688},
	url = {https://doi.org/10.1016/j.electacta.2022.141688},
	year = 2023,
	month = {jan},
	publisher = {Electrochimica Acta},
	volume = {439},
	pages = {141688},
	author = {Baptiste Py, Adeleke Maradesa, Francesco Ciucci},
	title = {Gaussian processes for the analysis of electrochemical impedance spectroscopy data: Prediction, filtering, and active learning},
	journal = {Electrochimica Acta}
}
```

## References
1. Ciucci, F. (2020). The Gaussian process Hilbert transform (GP-HT): testing the Ccnsistency of electrochemical impedance spectroscopy data. Journal of The Electrochemical Society, 167, 12, 126503. [doi.org/10.1149/1945-7111/aba937] (https://doi.org/10.1149/1945-7111/aba937)
2. Py, B., Maradesa, A., & Ciucci, F. (2023). Gaussian processes for the analysis of electrochemical impedance spectroscopy data: Prediction, filtering, and active learning. Electrochimica Acta, 439, 141688. [doi.org/10.1016/j.electacta.2022.141688] (https://doi.org/10.1016/j.electacta.2022.141688)
3. Liu, J., Wan, T. H., & Ciucci, F. (2020). A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores. Electrochimica Acta, 357, 136864. [doi.org/10.1016/j.electacta.2020.136864] (https://doi.org/10.1016/j.electacta.2020.136864)
4. Ciucci, F. (2019). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry, 13, 132-139. [doi.org/10.1016/j.coelec.2018.12.003](https://doi.org/10.1016/j.coelec.2018.12.003)
5. Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. [doi.org/10.1016/j.electacta.2014.09.058](https://doi.org/10.1016/j.electacta.2014.09.058)
6. Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. [doi.org/10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097)
7. Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. [doi.org/10.1016/j.electacta.2015.03.123](https://doi.org/10.1016/j.electacta.2015.03.123)
8. Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. [doi.org/10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050)
9. Liu, J., & Ciucci, F. (2019). The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. [doi.org/10.1016/j.electacta.2019.135316](https://doi.org/10.1016/j.electacta.2019.135316)
10. Liu, J., & Ciucci, F. (2020). The Deep-Prior distribution of relaxation times. Journal of The Electrochemical Society, 167(2), 026506.[10.1149/1945-7111/ab631a](https://iopscience.iop.org/article/10.1149/1945-7111/ab631a/meta)
11. Maradesa, A., Py, B., Quattrocchi, E., & Ciucci, F. (2022). The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes. Electrochimica Acta, 413, 140119. [doi.org/10.1016/j.electacta.2022.140119] (https://doi.org/10.1016/j.electacta.2022.140119)
12. Quattrocchi, E., Py, B., Maradesa, A., Meyer, Q., Zhao, C., & Ciucci, F. (2023) Deconvolution of electrochemical impedance spectroscopy data using the deep-neural-network-enhanced distribution of relaxation times. Electrochimica Acta, 439, 141499. [doi.org/10.1016/j.electacta.2022.141499] (https://doi.org/10.1016/j.electacta.2022.141499)
