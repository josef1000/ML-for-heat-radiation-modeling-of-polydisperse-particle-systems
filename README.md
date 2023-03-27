# ML-for-heat-radiation-modeling-of-polydisperse-particle-systems

* All Machine Learning-based models (except Random Forest-based regressors) discussed in publication J. Tausendschön, G. Stöckl, S. Radl, Machine Learning for heat radiation modeling of bi- and polydisperse particle systems including walls, Particuology. 74 (2023) 119–140. See https://doi.org/10.1016/j.partic.2022.05.011 .

* Random Forest-based regressor models can be downloaded here: https://cloud.tugraz.at/index.php/s/q9FymHBKxWQ7icY

* The four inputs of the models are: $(r_j / |\mathbf{x}\_{i-j}|, \ n_{pib,i-j,r_{max}}, \ r_j/r_i, \ \phi_{p,sys})$ 

* The output of the models is the view factor directly. However, for numerical reasons the target view factors in training were min-max normalized and subsequently the model output needs to be denormalized for usage.

* The radiative heat flux between two particles, where $\varepsilon\_{i-j}$ describes the view factor between emitting particle $i$ and absorbing particle $j$, can then be calculated based on the view factor by:

$$
\dot{Q}\_{rad,i-j}
  = \frac{\sigma\_{SB} \ (T_i^4-T_j^4)}{\dfrac{1-\epsilon_i}{A_i \ \epsilon_i}+\dfrac{1-\epsilon_j}{A_j \ \epsilon_j}+\dfrac{1}{\varepsilon\_{i-j} \ A_i}}
$$

*	The script usageDemo.py shows how to load and compile the saved models, normalize the input, get the prediction based on sample data and how to denormalize the prediction.

