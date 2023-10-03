import pandas as pd
import gurobipy as gp
import os
from .battery_optimisation import BatteryModel

# Constants
MARKET_COST_FACTOR = 1.2
NUM_INTERVALS_DAY = 48

class DataPreparation():
    def __init__(self, prosumer_binding_data_file, price_type='dispatch', dispatch_price_file=None):
        """Initialise DataPreparation class.
        
        Parameters:
            prosumer_binding_data_file (str): Path to prosumer aggregated binding data file.
            price_type (str, optional): Type of price ('dispatch' or 'pre_dispatch'). Defaults to 'dispatch'.
            dispatch_price_file (str, optional): Path to dispatch price file. Required if price_type is 'dispatch'.
        """
        self.prosumer_binding_data_file = prosumer_binding_data_file
        self.price_type = price_type
        if price_type == 'dispatch':
            # Ensure that dispatch_price_file is not None
            assert dispatch_price_file is not None, 'provide file localtion for dispatch price'
        self.dispatch_price_file = dispatch_price_file

    def _get_dispatch_price(self):
        """Read dispatch price from csv file.
        
        Returns:
            pd.DataFrame: Dataframe with dispatch prices.
        """
        df_dispatch_price = pd.read_csv(self.dispatch_price_file, low_memory=False, float_precision='round_trip', 
                                        index_col='SETTLEMENTDATE', parse_dates=True)
        # Convert $/MWh to $/kWh
        df_dispatch_price['RRP_NSW1'] = df_dispatch_price['RRP_NSW1'] / 1000
        df_dispatch_price['month'] = df_dispatch_price.index.month
        df_dispatch_price['day'] = df_dispatch_price.index.day
        df_dispatch_price['time_of_day'] = df_dispatch_price.index.time
        return df_dispatch_price

    def _compile_data_for_dispatch_price(self, df_prosumer_data, df_dispatch_price):
        """Add dispatch price to prosumer data by replacing the pd prices.
        
        Parameters:
            df_prosumer_data (pd.DataFrame): Prosumer data.
            df_dispatch_price (pd.DataFrame): Dispatch price data.
        
        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        df_prosumer_data.index = pd.to_datetime(df_prosumer_data.index)
        df_prosumer_data['month'] = df_prosumer_data.index.month
        df_prosumer_data['day'] = df_prosumer_data.index.day
        df_prosumer_data['time_of_day'] = df_prosumer_data.index.time

        df_prosumer_agg_binding = df_prosumer_data.reset_index().merge(df_dispatch_price, on=['month', 'day', 'time_of_day'])
        df_prosumer_agg_binding = df_prosumer_agg_binding.set_index('time') 
        # 'price' column is originally the pd price. Replace it with dispatch price
        df_prosumer_agg_binding['price'] = df_prosumer_agg_binding['RRP_NSW1']
        # Drop unnecessary columns
        drop_cols = ['month', 'day', 'time_of_day', 'RRP_NSW1']
        df_prosumer_agg_binding = df_prosumer_agg_binding.drop(drop_cols, axis=1)
        return df_prosumer_agg_binding

    def read_prosumer_binding_data(self):
        """Read prosumer binding data based on the price type.
        
        Returns:
            pd.DataFrame: Dataframe with prosumer binding data.
        """
        df_prosumer_agg_binding = pd.read_csv(self.prosumer_binding_data_file, low_memory=False, float_precision='round_trip', 
                                              index_col='time', parse_dates=True)
        if self.price_type == 'pre_dispatch':
            return df_prosumer_agg_binding
        elif self.price_type == 'dispatch':
            df_dispatch_price = self._get_dispatch_price()
            df_prosumer_agg_binding = self._compile_data_for_dispatch_price(df_prosumer_agg_binding, df_dispatch_price)
            return df_prosumer_agg_binding
        

class BatteryWithoutRhModel(BatteryModel):
    def __init__(self, optimisation_horizon):
        """Initialise BatteryWithoutRhModel class.
        
        Parameters:
            optimisation_horizon (list): Time intervals (from the sizing horizon) for optimisation.
        """
        super().__init__(optimisation_horizon)

    def _set_objective_function(self, fee_information, battery_spec):
        """Modify the objective function from BatteryModel because receding horizon is not used."""
        # Calculate different components of the objective function
        energy_cost = gp.quicksum(fee_information['spot_price'][t] * self.m_var['community_net_positive'][t] * MARKET_COST_FACTOR
                                  for t in self.optimisation_horizon)
        dnsp_cost = gp.quicksum(fee_information['LUoS'] * self.m_var['grid_charging_positive'][t]
                                for t in self.optimisation_horizon)
        battery_opex = gp.quicksum(fee_information['charging_fee'] * self.m_var['charging_energy_negative'][t]
                                   for t in self.optimisation_horizon)
        
        n_optimised_days = len(self.optimisation_horizon) / NUM_INTERVALS_DAY
        peak_demand_cost = fee_information['peak_charge'] * self.m_var['max_net_energy'] * n_optimised_days / battery_spec['time_resolution']
        # Get battery cost based on 10 years warranty period
        battery_cost = battery_spec['price'] * self.m_var['battery_capacity'] * n_optimised_days / (365 * battery_spec['warranty_period'])

        # Set the objective function to minimise the total cost
        self.model.setObjective(energy_cost + dnsp_cost + battery_opex + peak_demand_cost + battery_cost, 
                                gp.GRB.MINIMIZE)


class BatteryWithoutRhSizing:
    def __init__(self, parameter_handler, optimisation_horizon):
        """Initialise BatteryWithoutRhSizing class.
        
        Parameters:
            parameter_handler (object): Instance of OptimisationParameters from battery_optimisation file.
            optimisation_horizon (list): Time intervals (from the sizing horizon) for optimisation.
        """
        self.parameter_handler = parameter_handler
        self.optimisation_horizon = optimisation_horizon
        self.battery_model = BatteryWithoutRhModel(optimisation_horizon)
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

    def _get_optimisation_parameter(self):
        """Get optimisation parameters
        
        Returns:
            tuple: Contains aggregate DataFrame, battery, prosumer and fee specifications.
        """
        df_aggregate = self.parameter_handler.get_prosumer_agg_data(rolling_th='full')
        battery_spec = self.parameter_handler.get_battery_spec()
        prosumer_spec = self.parameter_handler.get_prosumer_spec(df_aggregate)
        fee_information = self.parameter_handler.get_price_spec(df_aggregate)
        return df_aggregate, battery_spec, prosumer_spec, fee_information

    def sizing_optimisation(self):
        """Perform sizing optimisation.
        
        Returns:
            pd.DataFrame: DataFrame containing optimised battery operation data.
        """
        # Define variables to fetch from optimisation model
        vars_to_get = ['battery_energy', 'charging_energy', 'charging_energy_positive', 
                       'charging_energy_negative', 'community_net_positive', 
                       'community_net_negative', 'grid_charging_positive']
        batt_oper_dict = {var: [] for var in vars_to_get}

        # Get constant values
        df_aggregate, battery_spec, prosumer_spec, fee_information = self._get_optimisation_parameter()
        # Run the optimisation model
        model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                  initial_soc=0, current_max_net_energy=0)

        # Fetch the optimisation results
        batt_oper_dict['time'] = df_aggregate.index
        for var in vars_to_get:
            batt_oper_dict[var] = [model.getVarByName(f'{var}[{t}]').x for t in self.optimisation_horizon]
        batt_oper_dict['capacity'] = model.getVarByName('battery_capacity').x

        self.df_battery_optimised = self._compile_battery_optimised_data(batt_oper_dict)
        return self.df_battery_optimised
    
    def save_binding_data(self, folder_name, price_type):
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
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_{price_type}_without_rh_sizing.csv')