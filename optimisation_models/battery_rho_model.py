import pandas as pd
import numpy as np
import gurobipy as gp
import os
import glob
from .battery_optimisation import BatteryModel

class DataPreparation():
    def __init__(self, prosumer_agg_rho_folder):
        self.prosumer_agg_rho_folder = prosumer_agg_rho_folder
        self.interval_counts = 0
        
    def read_monthly_prosumer_aggregate(self, month):
        prosumer_agg_rho = []
        month_agg_folder = f'{self.prosumer_agg_rho_folder}/month_{month}'
        num_files = len(glob.glob(f'{month_agg_folder}/*.csv'))
        for interval in range(num_files):
            agg_file_name = f'{month_agg_folder}/aggregate_interval_{interval}_pd_price.csv'
            df_intervally_aggregate = pd.read_csv(agg_file_name, low_memory=False, 
                                                  float_precision='round_trip', index_col='time')
            df_intervally_aggregate['rolling_th'] = interval + self.interval_counts
            df_intervally_aggregate.index = pd.to_datetime(df_intervally_aggregate.index)
            prosumer_agg_rho.append(df_intervally_aggregate)
        self.interval_counts += num_files
        df_prosumer_agg_rho = pd.concat(prosumer_agg_rho)
        return df_prosumer_agg_rho

    def read_prosumer_aggregate_rho(self, month_list):
        dfs_monthly_prosumer_agg = []
        for month in month_list:
            df_prosumer_agg_rho = self.read_monthly_prosumer_aggregate(month)
            dfs_monthly_prosumer_agg.append(df_prosumer_agg_rho)
        df_prosumer_agg_rho = pd.concat(dfs_monthly_prosumer_agg)
        return df_prosumer_agg_rho
    

class BatteryRho():
    def __init__(self, parameter_handler, optimisation_horizon):
        self.parameter_handler = parameter_handler
        self.battery_model = BatteryModel(optimisation_horizon)
        self.batt_oper_dict = None
        self.df_battery_optimised = None
    
    def _compile_battery_optimised_data(self, battery_spec):
        df_battery = pd.DataFrame(data=self.batt_oper_dict)
        df_battery = df_battery.set_index('time')
        df_battery.index = pd.to_datetime(df_battery.index)
        # Ensure that the battery capacity is not None. If None, raise an error
        if battery_spec['capacity'] is None:
            raise ValueError('Battery capacity is not available')
        df_battery['capacity'] = battery_spec['capacity']
        return df_battery

    def _get_optimisation_parameter(self, rolling_th):
        df_aggregate = self.parameter_handler.get_prosumer_agg_data(rolling_th)
        battery_spec = self.parameter_handler.get_battery_spec()
        prosumer_spec = self.parameter_handler.get_prosumer_spec(df_aggregate)
        fee_information = self.parameter_handler.get_price_spec(df_aggregate)
        return df_aggregate, battery_spec, prosumer_spec, fee_information

    def rolling_optimisation(self, n_receding_horizons):
        vars_to_get = ['battery_energy', 'charging_energy', 'charging_energy_positive', 
                       'charging_energy_negative', 'community_net_positive', 
                       'community_net_negative', 'grid_charging_positive']
        self.batt_oper_dict = {var: [] for var in vars_to_get}
        self.batt_oper_dict['time'] = []

        for interval in range(n_receding_horizons):
            if interval % 500 == 0:
                print('INTERVAL:', interval)

            ### Get constant values
            df_aggregate, battery_spec, prosumer_spec, fee_information = self._get_optimisation_parameter(interval)

            initial_SOC = 0 if interval == 0 else self.batt_oper_dict['battery_energy'][-1]
            current_max_net = 0 if interval == 0 else np.max(self.batt_oper_dict['community_net_positive'])
            model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                      initial_SOC=initial_SOC, current_max_net_energy=current_max_net)

            # Get only the realised values which are the first values of the decision variables
            self.batt_oper_dict['time'].append(df_aggregate.index[0])
            for var in vars_to_get:
                self.batt_oper_dict[var].append(model.getVarByName(f'{var}[0]').x)

        self.df_battery_optimised = self._compile_battery_optimised_data(battery_spec)
        return self.df_battery_optimised
    
    def save_binding_data(self, folder_name):
        if self.df_battery_optimised is None:
            raise ValueError('No data to save. Please run the rolling optimisation first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        capacity = self.df_battery_optimised['capacity'].iloc[0]
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_rho_model.csv')
        
