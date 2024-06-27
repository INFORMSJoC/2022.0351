[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Clustering then Estimation of Spatio-Temporal Self-Exciting Processes

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper
[Clustering then Estimation of Spatio-Temporal Self-Exciting Processes](https://doi.org/10.1287/ijoc.2022.0314) by H. Zhang, D. Zhan, J. Anderson, R. Righter, and Z. Zheng.


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0351

https://doi.org/10.1287/ijoc.2022.0351.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{zhang2024clustering,
  author =        {Zhang, Haoting and Zhan, Donglin and Anderson, James and Righter, Rhonda and Zheng, Zeyu},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Clustering then Estimation of Spatio-Temporal Self-Exciting Processes}},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0351.cd},
  url =           {https://github.com/INFORMSJoC/2022.0351},
  note =          {Available for download at https://github.com/INFORMSJoC/2022.0351},
}  
```

## Description

This directory contains the code for the _Clustering then Estimation of Spatio-Temporal Self-Exciting Processes (CTE)_ algorithm.

This directory contains the folders `src` and `data`:
- `src`: includes the source code of the paper. This code is organized as follows:
  - `src/table2`: contains for methods for Table 2 (General Comparisons).
  - `src/table3`: contains for methods for Table 3 (Tree-Edit Distance and Likelihood of Mis-specified Models).
  - `src/table4`: contains for methods for Table 4 (CTE-DBSCAN with Different Clustering Hyper-parameter).
  - `src/table5`: contains for implementation for Table 5 (Parameter Estimation with Different Triggering Functions).
  - `src/table6`: contains for implementation for Table 6 of real-world dataset.
  - `src/figure2`: contains for error ratio visualization for Figure 2.
- `data`: contains the raw datasets for Table 6.

## Dependencies
The following Python (3.8) packages are required to run this code:
- `minisom 2.3.1`
- `numpy 1.18.0`
- `pandas 1.1.3`
- `scikit-learn 0.23.1`
- `st-dbscan 0.2.2`
- `zss 1.2.0`
- `scipy 1.5.2`

## Data

In Section 5.2, we illustrate the effectiveness of the CTE method by experimenting with four real-world datasets, which can be found at:
[https://www.kaggle.com/mchirico/montcoalert/notebooks](https://www.kaggle.com/mchirico/montcoalert/notebooks),
[https://www.kaggle.com/blackecho/italy-earthquakes](https://www.kaggle.com/blackecho/italy-earthquakes),
[https://www.kaggle.com/datasets/sujan97/citibike-system-data](https://www.kaggle.com/datasets/sujan97/citibike-system-data),
[https://archive.ics.uci.edu/dataset/352/online+retail\#dataset](https://archive.ics.uci.edu/dataset/352/online+retail\#dataset).

## Run experiments
This section includes the code for all the experiment results in Section 5.

Fill the configuration parameters in `table2.py `, `table3.py `, `table4.py `, `table5.py `, and `table6.py `.

**_General Example_**

In `table2.py `, to run the code for setting $\mu = 0.02 , \alpha = 10, \beta = 5, \sigma_{x} = 0.2, \sigma_{y}=0.2$ with _CTE_DBSCAN_ algorithm, we need to set the following parameters:

```python
cte_config['lam'] = 0.02  # mu
cte_config['alpha'] = 10
cte_config['beta'] = 5
cte_config['sigx'] = 0.2 # sigma_x
cte_config['sigy'] = 0.2 # sigma_y

cte_config['cte_'] = True # cte

cte_config['cluster'] = 'd' #  cte_config['cluster']  d - dbscan , st - stdbscan ,  h - agglomerative , s - som
```

with filling other parameters (e.g. initial_guess, sampler_rate etc.).

No need to change the predefined parameters (e.g. simulation).