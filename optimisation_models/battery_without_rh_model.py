import pandas as pd
import numpy as np
import gurobipy as gp
import os
from .battery_optimisation import BatteryModel

MARKET_COST_FACTOR = 1.2
NUM_INTERVALS_DAY = 48

class DataPreparation():
    def __init__(self, prosumer_binding_data_file, price_type='dispatch', dispatch_price_file=None):
        self.prosumer_binding_data_file = prosumer_binding_data_file
        self.price_type = price_type
        if price_type == 'dispatch':
            # Ensure that dispatch_price_file is not None
            assert dispatch_price_file is not None, 'provide file localtion for dispatch price'
        self.dispatch_price_file = dispatch_price_file

    def _get_dispatch_price(self):
        df_dispatch_price = pd.read_csv(self.dispatch_price_file, low_memory=False, float_precision='round_trip', 
                                        index_col='SETTLEMENTDATE', parse_dates=True)
        # Convert $/MWh to $/kWh
        df_dispatch_price['RRP_NSW1'] = df_dispatch_price['RRP_NSW1'] / 1000
        df_dispatch_price['month'] = df_dispatch_price.index.month
        df_dispatch_price['day'] = df_dispatch_price.index.day
        df_dispatch_price['time_of_day'] = df_dispatch_price.index.time
        return df_dispatch_price

    def _compile_data_for_dispatch_price(self, df_prosumer_data, df_dispatch_price):
        df_prosumer_data.index = pd.to_datetime(df_prosumer_data.index)
        df_prosumer_data['month'] = df_prosumer_data.index.month
        df_prosumer_data['day'] = df_prosumer_data.index.day
        df_prosumer_data['time_of_day'] = df_prosumer_data.index.time

        df_prosumer_agg_binding = df_prosumer_data.reset_index().merge(df_dispatch_price, on=['month', 'day', 'time_of_day'])
        df_prosumer_agg_binding = df_prosumer_agg_binding.set_index('time')
        df_prosumer_agg_binding['price'] = df_prosumer_agg_binding['RRP_NSW1']
        # Drop RRP_NSW1 column
        drop_cols = ['month', 'day', 'time_of_day', 'RRP_NSW1']
        df_prosumer_agg_binding = df_prosumer_agg_binding.drop(drop_cols, axis=1)
        return df_prosumer_agg_binding

    def read_prosumer_binding_data(self):
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
        super().__init__(optimisation_horizon)

    def _set_objective_function(self, fee_information, battery_spec):   
        
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

        # Set the objective function to maximize the total profit
        self.model.setObjective(energy_cost + dnsp_cost + battery_opex + peak_demand_cost + battery_cost, 
                                gp.GRB.MINIMIZE)


class BatteryWithoutRhSizing:
    def __init__(self, parameter_handler, optimisation_horizon):
        self.parameter_handler = parameter_handler
        self.optimisation_horizon = optimisation_horizon
        self.battery_model = BatteryWithoutRhModel(optimisation_horizon)
        self.df_battery_optimised = None

    def _compile_battery_optimised_data(self, batt_oper_dict):
        df_battery = pd.DataFrame(data=batt_oper_dict)
        df_battery = df_battery.set_index('time')
        df_battery.index = pd.to_datetime(df_battery.index)
        return df_battery

    def _get_optimisation_parameter(self):
        df_aggregate = self.parameter_handler.get_prosumer_agg_data(rolling_th='full')
        battery_spec = self.parameter_handler.get_battery_spec()
        prosumer_spec = self.parameter_handler.get_prosumer_spec(df_aggregate)
        fee_information = self.parameter_handler.get_price_spec(df_aggregate)
        return df_aggregate, battery_spec, prosumer_spec, fee_information

    def sizing_optimisation(self):
        vars_to_get = ['battery_energy', 'charging_energy', 'charging_energy_positive', 
                       'charging_energy_negative', 'community_net_positive', 
                       'community_net_negative', 'grid_charging_positive']
        batt_oper_dict = {var: [] for var in vars_to_get}

        # Get constant values
        df_aggregate, battery_spec, prosumer_spec, fee_information = self._get_optimisation_parameter()
        model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                  initial_SOC=0, current_max_net_energy=0)

        # Get only the realised values which are the first values of the decision variables
        batt_oper_dict['time'] = df_aggregate.index
        for var in vars_to_get:
            batt_oper_dict[var] = [model.getVarByName(f'{var}[{t}]').x for t in self.optimisation_horizon]
        batt_oper_dict['capacity'] = model.getVarByName('battery_capacity').x

        self.df_battery_optimised = self._compile_battery_optimised_data(batt_oper_dict)
        return self.df_battery_optimised
    
    def save_binding_data(self, folder_name, price_type):
        if self.df_battery_optimised is None:
            raise ValueError('No data to save. Please run the rolling optimisation first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        capacity = int(self.df_battery_optimised['capacity'].iloc[0])
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_{price_type}_without_rh_sizing.csv')