import pandas as pd
import numpy as np
import gurobipy as gp
import os
import glob

OPTIMISATION_HORIZON = np.arange(32)
MARKET_COST_FACTOR = 1.2

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
            df_intervally_aggregate['run_time'] = df_intervally_aggregate.index[0]
            df_intervally_aggregate['interval'] = interval + self.interval_counts
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

class OptimisationParameters:
    def __init__(self, df_prosumer_agg_rho, battery_capacity):
        self.df_prosumer_agg_rho = df_prosumer_agg_rho
        self.battery_capacity = battery_capacity

    def get_battery_spec(self):
        """
        Battery specs
        Output a dictionary
        """

        battery_spec = dict()
        battery_spec['lower_SOC'] = 0
        battery_spec['upper_SOC'] = 1.0
        battery_spec['capacity'] = self.battery_capacity # [kWh]
        battery_duration = 2 # [h]
        battery_spec['charging_power'] = battery_spec['capacity'] / battery_duration
        battery_spec['efficiency'] = 0.9 # Round-trip efficiency
        battery_spec['time_resolution'] = 0.5 # [h]
        
        return battery_spec
    
    def _extract_rolling_aggregate(self, rolling_th):
        df_aggregate = self.df_prosumer_agg_rho[self.df_prosumer_agg_rho['interval'] == rolling_th]
        return df_aggregate

    def get_prosumer_spec(self, rolling_th):
        """
        Prosumer specs
        Output a dictionary
        """
        df_aggregate = self._extract_rolling_aggregate(rolling_th)
        columns_of_interest = ['run_time', 'net_positive', 'export_energy', 'import_energy']

        prosumer_spec = {}
        for column in columns_of_interest:
            prosumer_spec[column] = df_aggregate[column]

        return prosumer_spec
    
    def get_price_spec(self, rolling_th):
        """
        Get the list of spot price and the maximum price in the current rolling horizon.
        """
        df_aggregate = self._extract_rolling_aggregate(rolling_th)
        # Get the spot price and the maximum price in the current rolling horizon
        fee_information = dict()
        fee_information['spot_price'] = df_aggregate['pd_price']
        fee_information['charging_fee'] = 0.032 # $/kWh
        fee_information['LUoS'] = 0.016098 # $/kWh
        fee_information['peak_charge'] = 0.331069 # $/kW/day
        
        return fee_information

class _BatteryModel():
    def __init__(self):
        self.model = None
        self.m_var = None
    
    def _set_model_var(self):
        
        def add_model_vars(*args, **kwargs):
            """Helper function to add variables to the model and store them in m_var."""
            var_name = kwargs['name']
            self.m_var[var_name] = self.model.addVars(*args, **kwargs)

        # Create a dict of model variables
        self.m_var = dict()

        # Battery variables
        add_model_vars(OPTIMISATION_HORIZON, name='charging_energy', lb=-gp.GRB.INFINITY)
        add_model_vars(OPTIMISATION_HORIZON, name='battery_energy')
        add_model_vars(OPTIMISATION_HORIZON, name='charging_energy_positive')
        add_model_vars(OPTIMISATION_HORIZON, name='charging_energy_negative')

        # Separate local net
        add_model_vars(OPTIMISATION_HORIZON, name='community_net_positive')
        add_model_vars(OPTIMISATION_HORIZON, name='community_net_negative')
        add_model_vars(OPTIMISATION_HORIZON, name='community_net_binary', vtype=gp.GRB.BINARY)

        # Separate energy charged from utility grid
        add_model_vars(OPTIMISATION_HORIZON, name='grid_charging_positive')

        # Maximum net energy
        self.m_var['max_net_energy'] = self.model.addVar(name='max_net_energy')


    def _set_battery_constraint(self, initial_SOC, battery_spec):
    
        # Charging constraints
        # Multiple by time resolution (0.5) representing half an hour
        self.model.addConstrs(self.m_var['charging_energy'][t] <= battery_spec['time_resolution'] * battery_spec['charging_power']
                              for t in OPTIMISATION_HORIZON)
        self.model.addConstrs(self.m_var['charging_energy'][t] >= -battery_spec['time_resolution'] * battery_spec['charging_power']
                              for t in OPTIMISATION_HORIZON)
        # Complemetarity constraint for battery charging
        self.model.addConstrs(self.m_var['charging_energy'][t] == self.m_var['charging_energy_positive'][t] - self.m_var['charging_energy_negative'][t] 
                              for t in OPTIMISATION_HORIZON)
        
        # SOC determination
        self.model.addConstrs(self.m_var['battery_energy'][t] == initial_SOC + 
                              gp.quicksum(self.m_var['charging_energy_positive'][j] - self.m_var['charging_energy_negative'][j] / battery_spec['efficiency'] 
                                          for j in range(t + 1)) for t in OPTIMISATION_HORIZON)

        # Ending SOC of the receding horizon is equal the initial SOC
        self.model.addConstr(self.m_var['battery_energy'][len(OPTIMISATION_HORIZON) - 1] == initial_SOC)
        
        # Define bounbdaries for SOC
        self.model.addConstrs(self.m_var['battery_energy'][t] >= battery_spec['lower_SOC'] * battery_spec['capacity']
                              for t in OPTIMISATION_HORIZON)
        self.model.addConstrs(self.m_var['battery_energy'][t] <= battery_spec['upper_SOC'] * battery_spec['capacity']
                              for t in OPTIMISATION_HORIZON)
    
    def _set_net_energy_constraint(self, prosumer_spec):
        big_M = 1000

        self.model.addConstrs(prosumer_spec['net_positive'][t] - prosumer_spec['export_energy'][t]
                              + self.m_var['charging_energy'][t] == self.m_var['community_net_positive'][t] 
                              - self.m_var['community_net_negative'][t] for t in OPTIMISATION_HORIZON)

        # Complementarity constraint for community net energy
        self.model.addConstrs(self.m_var['community_net_positive'][t] <= big_M * (1 - self.m_var['community_net_binary'][t])
                              for t in OPTIMISATION_HORIZON)
        self.model.addConstrs(self.m_var['community_net_negative'][t] <= big_M * self.m_var['community_net_binary'][t]
                              for t in OPTIMISATION_HORIZON)

        self.model.addConstrs(self.m_var['charging_energy_positive'][t] - prosumer_spec['export_energy'][t] <=
                              self.m_var['grid_charging_positive'][t] for t in OPTIMISATION_HORIZON)
    
    def _set_peak_demand_constraint(self, current_max_net_energy):
        # Define maximum net energy
        self.model.addConstrs(self.m_var['max_net_energy'] >= self.m_var['community_net_positive'][t] 
                              for t in OPTIMISATION_HORIZON)
        self.model.addConstr(self.m_var['max_net_energy'] >= current_max_net_energy)
        
    def _set_objective_function(self, fee_information, battery_spec):   
        
        energy_cost = gp.quicksum(fee_information['spot_price'][t] * self.m_var['community_net_positive'][t] * MARKET_COST_FACTOR
                                  for t in OPTIMISATION_HORIZON)
        dnsp_cost = gp.quicksum(fee_information['LUoS'] * self.m_var['grid_charging_positive'][t]
                                for t in OPTIMISATION_HORIZON)
        battery_opex = gp.quicksum(fee_information['charging_fee'] * self.m_var['charging_energy_negative'][t]
                                   for t in OPTIMISATION_HORIZON)
        
        peak_demand_cost = fee_information['peak_charge'] * 365 * self.m_var['max_net_energy'] / battery_spec['time_resolution']

        # Set the objective function to maximize the total profit
        self.model.setObjective(energy_cost + dnsp_cost + battery_opex + peak_demand_cost, gp.GRB.MINIMIZE)
    
    def optimise_model(self, battery_spec, prosumer_spec, fee_information, 
                       initial_SOC, current_max_net_energy):
        
        ### Model declaration
        self.model = gp.Model('BatteryRho')
        self.model.setParam('OutputFlag', 0)
        
        ## Decision variables
        self._set_model_var()
        ## Constraints
        self._set_battery_constraint(initial_SOC, battery_spec)
        self._set_net_energy_constraint(prosumer_spec)
        self._set_peak_demand_constraint(current_max_net_energy)
        ## Objective function
        self._set_objective_function(fee_information, battery_spec)

        ## Run the optimization model
        self.model.optimize()

        return self.model

class BatteryRho:
    def __init__(self, parameter_handler):
        self.parameter_handler = parameter_handler
        self.battery_model = _BatteryModel()
        self.batt_oper_dict = None
        self.df_battery_optimised = None
    
    def _compile_battery_optimised_data(self, battery_spec):
        df_battery = pd.DataFrame(data=self.batt_oper_dict)
        df_battery = df_battery.set_index('time')
        df_battery.index = pd.to_datetime(df_battery.index)
        df_battery['capacity'] = battery_spec['capacity']
        return df_battery

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
            battery_spec = self.parameter_handler.get_battery_spec()
            prosumer_spec = self.parameter_handler.get_prosumer_spec(interval)
            fee_information = self.parameter_handler.get_price_spec(interval)

            initial_SOC = 0 if interval == 0 else self.batt_oper_dict['battery_energy'][-1]
            current_max_net = 0 if interval == 0 else np.max(self.batt_oper_dict['community_net_positive'])
            model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                      initial_SOC=initial_SOC, current_max_net_energy=current_max_net)

            # Get only the realised values which are the first values of the decision variables
            self.batt_oper_dict['time'].append(prosumer_spec['run_time'][0])
            for var in vars_to_get:
                self.batt_oper_dict[var].append(model.getVarByName(f'{var}[0]').x)

        self.df_battery_optimised = self._compile_battery_optimised_data(battery_spec)
        return self.df_battery_optimised
    
    def save_committed_data(self, folder_name):
        if self.df_battery_optimised is None:
            raise ValueError('No data to save. Please run the rolling optimisation first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        capacity = self.df_battery_optimised['capacity'].iloc[0]
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_rho_model.csv')
        
