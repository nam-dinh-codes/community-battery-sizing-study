# Community Battery Sizing Study Considering Receding Horizon Operation

## Introduction
This is the repository for a community battery storage (CBS) sizing study that considers receding horizon operation (RHO). A key part of this study is to develop an accurate CBS operation model that continuously adjusts to changes in power system forecasts. The uncertainties include wholesale spot prices and residential users' consumption. While the forecasts of spot prices can be obtained from the pre-dispatch prices published by AEMO, the forecasts of residential users' consumption are not available. Therefore, the study first considers a price-responsive behaviour model of residential elecitricity users to obtain the dynamic consumption forecasts of residential end users. The study then uses the price and consumption forecasts to determine the optimal CBS operation plan in a receding horizon manner. There exists different battery sizing studies in the literature, however, most of them do not consider the RHO operation. The study aims to compare the CBS sizing results with and without RHO operation.

## Usage

### End-users model
The end-users model is implemented in the `optimisation_models/prosumer_rho_model.py` file and can be run using the `prosumer_rolling_operation.ipynb` file. The output from the model is the varying consumption behaviour over time.
<p align="center">
<img src="data/figures/end_user_rho.jpg" alt="End users RHO flowchart" width="500">
</p>

### CBS operation model
The CBS operation model is implemented in the `optimisation_models/battery_rho_model.py` file and can be run using the `battery_rolling_operation.ipynb` file. The model considers the dynamic consumption forecasts of residential users and the dynamic wholesale spot prices forecasts. The output from the model is the optimised CBS operation plan over time.
<p align="center">
<img src="data/figures/cbs_rho.jpg" alt="Battery RHO flowchart" width="500">
</p>
The global optimal battery capacity can be found by exhaustively searching for the capacity that minimises the ground truth cost [To be updated]

### CBS sizing model without receding horizon
A common battery sizing approach is to assume a perfect prediction of uncertain parameters and solve a planning problem over the entire sizing horizon. Additionally, to explore the impact of forecast prices, we replace the actual prices with 30-minute look-ahead pre-dispatch prices. This sizing model is implemented in `optimisation_models/battery_without_rh_model.py` and can be run using the `battery_without_rh_sizing.ipynb`. 

### CBS sizing model with coupled receding horizons
The CBS sizing model with coupled receding horizons simultaneously considers all receding horizons in one optimisation problem. The model is implemented in `optimisation_models/battery_coupled_rh_model.py` and can be run using the `battery_with_rh_sizing.ipynb`.