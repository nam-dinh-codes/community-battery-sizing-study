import pandas as pd
import numpy as np
import os
import glob
from .battery_optimisation import BatteryModel

class DataPreparation():
    """Handles data preparation tasks."""

    def __init__(self, prosumer_agg_rho_folder):
        """Initialise DataPreparation class.
        
        Parameters:
            prosumer_agg_rho_folder (str): Folder containing prosumer aggregate data.
        """
        self.prosumer_agg_rho_folder = prosumer_agg_rho_folder
        self.interval_counts = 0
        
    def read_monthly_prosumer_aggregate(self, month):
        """Read monthly prosumer aggregate data.
        
        Parameters:
            month (int): The month for which data is to be read.
        
        Returns:
            pd.DataFrame: Concatenated DataFrame of monthly prosumer aggregated data.
        """
        prosumer_agg_rho = []
        # Construct folder path based on month
        month_agg_folder = f'{self.prosumer_agg_rho_folder}/month_{month}'
        num_files = len(glob.glob(f'{month_agg_folder}/*.csv'))
        # Loop over each interval (file) to read and aggregate data
        for interval in range(num_files):
            agg_file_name = f'{month_agg_folder}/aggregate_interval_{interval}_pd_price.csv'
            df_intervally_aggregate = pd.read_csv(agg_file_name, low_memory=False, 
                                                  float_precision='round_trip', index_col='time')
            # Give cumulative interval number (rolling_th) for each receding horizon
            df_intervally_aggregate['rolling_th'] = interval + self.interval_counts
            df_intervally_aggregate.index = pd.to_datetime(df_intervally_aggregate.index)
            prosumer_agg_rho.append(df_intervally_aggregate)
        # Update interval count
        self.interval_counts += num_files
        df_prosumer_agg_rho = pd.concat(prosumer_agg_rho)
        return df_prosumer_agg_rho

    def read_prosumer_aggregate_rho(self, month_list):
        """Read aggregate data for multiple months.
        
        Parameters:
            month_list (list): List of months for which to read data.
        
        Returns:
            pd.DataFrame: Concatenated DataFrame of multi-month data.
        """
        dfs_monthly_prosumer_agg = []
        for month in month_list:
            df_prosumer_agg_rho = self.read_monthly_prosumer_aggregate(month)
            dfs_monthly_prosumer_agg.append(df_prosumer_agg_rho)
        df_prosumer_agg_rho = pd.concat(dfs_monthly_prosumer_agg)
        return df_prosumer_agg_rho
    

class BatteryRho():
    """Handles battery receding horizon operation."""
    def __init__(self, parameter_handler, optimisation_horizon):
        """Initialise BatteryRho class.
        
        Parameters:
            parameter_handler (object): Instance of OptimisationParameters from battery_optimisation file.
            optimisation_horizon (int): Lookahead intervals for each receding horizon.
        """
        self.parameter_handler = parameter_handler
        self.battery_model = BatteryModel(optimisation_horizon)
        self.df_battery_optimised = None # Placeholder for optimised battery data
    
    def _compile_battery_optimised_data(self, batt_oper_dict):
        """Compile optimised battery operation data into a DataFrame.
        
        Parameters:
            batt_oper_dict (dict): Dictionary containing battery operation data.
        
        Returns:
            pd.DataFrame: Dataframe containing optimized battery operation data.
        """
        df_battery = pd.DataFrame(data=batt_oper_dict)
        df_battery = df_battery.set_index('time')
        df_battery.index = pd.to_datetime(df_battery.index)
        return df_battery

    def _get_optimisation_parameter(self, rolling_th):
        """Get optimisation parameters for a given time interval.
        
        Parameters:
            rolling_th (int): The current rolling time horizon index.
        
        Returns:
            tuple: Contains aggregate DataFrame, battery, prosumer and fee specifications.
        """
        df_aggregate = self.parameter_handler.get_prosumer_agg_data(rolling_th)
        battery_spec = self.parameter_handler.get_battery_spec()
        prosumer_spec = self.parameter_handler.get_prosumer_spec(df_aggregate)
        fee_information = self.parameter_handler.get_price_spec(df_aggregate)
        return df_aggregate, battery_spec, prosumer_spec, fee_information

    def rolling_optimisation(self, n_receding_horizons):
        """Perform rolling (receding) horizon optimization.
        
        Parameters:
            n_receding_horizons (int): Number of receding horizons to consider.
        
        Returns:
            pd.DataFrame: DataFrame containing optimised battery operation data.
        """
        # Define variables to fetch from optimisation model
        vars_to_get = ['battery_energy', 'charging_energy', 'charging_energy_positive', 
                       'charging_energy_negative', 'community_net_positive', 
                       'community_net_negative', 'grid_charging_positive']
        batt_oper_dict = {var: [] for var in vars_to_get}
        batt_oper_dict['time'] = []

        # Loop over each receding horizon to perform receding optimisation
        for interval in range(n_receding_horizons):
            # Print to show progress
            if interval % 500 == 0:
                print('INTERVAL:', interval)

            ### Get constant values
            df_aggregate, battery_spec, prosumer_spec, fee_information = self._get_optimisation_parameter(interval)
            initial_soc = 0 if interval == 0 else batt_oper_dict['battery_energy'][-1]
            current_max_net = 0 if interval == 0 else np.max(batt_oper_dict['community_net_positive'])
            # Run the optimisation model
            model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                      initial_soc=initial_soc, current_max_net_energy=current_max_net)

            # Fetch the optimisation results
            batt_oper_dict['time'].append(df_aggregate.index[0])
            for var in vars_to_get:
                batt_oper_dict[var].append(model.getVarByName(f'{var}[0]').x)

        batt_oper_dict['capacity'] = model.getVarByName('battery_capacity').x
        self.df_battery_optimised = self._compile_battery_optimised_data(batt_oper_dict)
        return self.df_battery_optimised
    
    def save_binding_data(self, folder_name):
        """Save the optimised battery operation data.
        
        Parameters:
            folder_name (str): Folder where the data should be saved.
        """
        if self.df_battery_optimised is None:
            raise ValueError('No data to save. Please run the rolling optimisation first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        capacity = int(round(self.df_battery_optimised['capacity'].iloc[0], 0))
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_rho_model.csv')