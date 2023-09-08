import pandas as pd
import numpy as np
import gurobipy as gp
import os
import glob

OPTIMISATION_HORIZON = np.arange(32)
REBOUND_HORIZON = np.arange(12)
PERCENTAGE_FLEXIBILITY = 0.5
HYPERBOLIC_CONVERGENCE = 0.2
N_PIECEWISE_LINEAR = 11
MARKET_COST_FACTOR = 1.2

class DataPreparation:
    def __init__(self, prosumer_file, pd_price_file):
        self.prosumer_file = prosumer_file
        self.pd_price_file = pd_price_file

    @staticmethod
    def limit_solar_prosumers(df_solar_prosumer, solar_prosumer_percentage):
        # Only allow about 50% of prosumers with rooftop solar
        df_max_solar = df_solar_prosumer.groupby(df_solar_prosumer.index).agg({'solar': 'max'})
        low_solar_prosumer = df_max_solar[df_max_solar['solar'] <= df_max_solar['solar'].quantile(solar_prosumer_percentage)].index

        for pro in low_solar_prosumer:
            df_solar_prosumer.loc[df_solar_prosumer.index == pro, 'solar'] = 0
        # Recalculate net energy after removing solar
        df_solar_prosumer['net energy'] = df_solar_prosumer['consumption'] - df_solar_prosumer['solar']

        return df_solar_prosumer

    def get_prosumer_data(self, month):
        df_raw_prosumer = pd.read_csv(self.prosumer_file, low_memory=False, 
                                      float_precision='round_trip', index_col='dataid')
        df_raw_prosumer['time'] = pd.to_datetime(df_raw_prosumer['time'])
        # Reduce the number of prosumers to 50% of the original number
        df_prosumer = self.limit_solar_prosumers(df_raw_prosumer.copy(), solar_prosumer_percentage=0.5)
        # Get prosumer data for the given month
        df_prosumer = df_prosumer[df_prosumer['time'].dt.month.isin([month])]
        return df_prosumer
    
    def get_pd_price_data(self, month):
        df_pd_price = pd.read_csv(self.pd_price_file, low_memory=False, float_precision='round_trip', 
                                  index_col='run_time', parse_dates=True)
        # Convert from $/MWh to $/kWh
        df_pd_price['RRP'] = np.around(df_pd_price['RRP'] / 1000, 5)
        # Get run_time for 2021
        df_pd_price = df_pd_price.loc['2021-01-01 00:00:00':'2021-12-31 23:30:00']
        # Get price data for the given month
        df_pd_price = df_pd_price[df_pd_price.index.month.isin([month])]
        return df_pd_price


class _OptimisationParameters:
    def __init__(self, df_prosumer, df_price):
        self.df_prosumer = df_prosumer
        self.df_price = df_price
        self.prosumer_list = sorted(set(self.df_prosumer.index))

    def get_prosumer_spec(self, rolling_th):
        """
        Extract prosumer specifications for a given rolling threshold.
        Returns a dictionary with prosumer specifications.
        """

        num_intervals = len(OPTIMISATION_HORIZON)
        df_rolling = self.df_prosumer.reset_index().pivot(index='time', columns='dataid')[rolling_th: rolling_th + num_intervals]

        prosumer_spec = {
            'elasticity': df_rolling['elasticity'],
            'df_consumption': df_rolling['consumption'],
            'df_net': df_rolling['net energy'],
            'df_solar': df_rolling['solar'],
            'import_charges': df_rolling['import_charges'],     # Get network charges
            'export_charges': df_rolling['export_charges'],     # Get network charges
            'export_limit': df_rolling['export_limit'],
            'discount_factor': df_rolling['discount_factor'],   # Get discount factor of time inconsistency
            'min_consumption': (1 - PERCENTAGE_FLEXIBILITY) * df_rolling['consumption'],    # Prosumers can vary per_flexbility % of the originally expected consumption
            'max_consumption': (1 + PERCENTAGE_FLEXIBILITY) * df_rolling['consumption']     # Prosumers can vary per_flexbility % of the originally expected consumption
        }

        return prosumer_spec
    
    def prosumer_comfort_piecewise(self, prosumer_spec, reference_price):
        """
        Generate piecewise consumption data for each prosumer over the optimisation horizon.
        """
        
        num_prosumers, num_intervals = len(self.prosumer_list), len(OPTIMISATION_HORIZON)
        
        step = np.zeros((num_prosumers, num_intervals, N_PIECEWISE_LINEAR))
        step_range = np.zeros((num_prosumers, num_intervals))
        y_value = np.zeros((num_prosumers, num_intervals, N_PIECEWISE_LINEAR))
        slope = np.zeros((num_prosumers, num_intervals, N_PIECEWISE_LINEAR-1))

        # Loop over time intervals and prosumers
        for t in OPTIMISATION_HORIZON:
            for i in self.prosumer_list:
                # Calculations related to time inconsistency and elasticity
                discount_factor = prosumer_spec['discount_factor'][i][t]
                hyperpolic_discount = (1 - HYPERBOLIC_CONVERGENCE) / (1 + discount_factor * t) + HYPERBOLIC_CONVERGENCE
                min_consumption = prosumer_spec['min_consumption'][i][t] - prosumer_spec['df_consumption'][i][t]
                max_consumption = prosumer_spec['max_consumption'][i][t] - prosumer_spec['df_consumption'][i][t]

                # Calculating step values and ranges
                step[i, t] = np.around(np.linspace(min_consumption, max_consumption, num=N_PIECEWISE_LINEAR), 7)
                step_range[i, t] = step[i, t, 1] - step[i, t, 0]

                # Calculating y-values and slopes
                elasticity, df_cons = prosumer_spec['elasticity'][i][t], prosumer_spec['df_consumption'][i][t]
                for s in range(N_PIECEWISE_LINEAR):
                    y_value[i, t, s] = (0 if df_cons == 0 else
                                        reference_price * step[i, t, s] *
                                        (1 + step[i, t, s] / (2 * elasticity * df_cons)) * hyperpolic_discount)
                for s in range(N_PIECEWISE_LINEAR-1):
                    slope[i, t, s] = (0 if (df_cons == 0 or step_range[i, t] == 0) else
                                      (y_value[i, t, s+1] - y_value[i, t, s]) / step_range[i, t])

        return {
            'step': step,
            'step_range': step_range,
            'slope': slope
        }

    def get_price_spec(self, rolling_th):
        """
        Get the list of spot price and the maximum price in the current rolling horizon.
        """

        rolling_time = self.df_prosumer.iloc[rolling_th]['time']

        # Create a mask for selecting rows with the same month, day, hour, and minute as the rolling_time
        matching_time_mask = (
            (self.df_price.index.month == rolling_time.month) &
            (self.df_price.index.day == rolling_time.day) &
            (self.df_price.index.hour == rolling_time.hour) &
            (self.df_price.index.minute == rolling_time.minute)
        )
        # Extract prices that match the time
        wm_price = self.df_price.loc[matching_time_mask, 'RRP']
        
        # Get the spot price and the reference price in the current rolling horizon
        spot_price = wm_price[:len(OPTIMISATION_HORIZON)]
        reference_price = spot_price.max()
        
        return spot_price, reference_price
    

class ProsumerModel:
    def __init__(self, df_prosumer, df_price):
        """
        Initialize the Model Building class.
        """
        self.parameter_handler = _OptimisationParameters(df_prosumer, df_price)
        self.prosumer_list = self.parameter_handler.prosumer_list
        self.model = None
        self.m_var = None

    def _set_model_var(self):
        """
        Create a dictionary of model variables.

        Returns:
            dict: Dictionary of model variables.
        """

        def add_model_vars(*args, **kwargs):
            """Helper function to add variables to the model and store them in m_var."""
            var_name = kwargs['name']
            self.m_var[var_name] = self.model.addVars(*args, **kwargs)

        self.m_var = dict()

        # Consumption related variables
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='consumption')
        add_model_vars(N_PIECEWISE_LINEAR - 1, self.prosumer_list, OPTIMISATION_HORIZON, name='pwl_consumption')
        add_model_vars(self.prosumer_list, name='consumption_deviation', lb=-gp.GRB.INFINITY)

        # Net energy variables
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='net_positive')
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='export_energy')
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='net_binary', vtype=gp.GRB.BINARY)
        add_model_vars(OPTIMISATION_HORIZON, name='local_net', lb=-gp.GRB.INFINITY)

        # Solar energy related variables
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='credit_usage')
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='cumulative_solar_credit')
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='used_solar')

        # Consumption post solar credits (import energy from the grid)
        add_model_vars(self.prosumer_list, OPTIMISATION_HORIZON, name='import_energy')
    
    def _set_consumption_constraint(self, prosumer_spec, piecewise_dict, previous_consumption_deviation):
        """
        Define prosumer constraints.

        Args:
            prosumer_spec (dict): Prosumer specifications.
            piecewise_dict (dict): Piecewise linear representations.
            previous_consumption_deviation (float): Previous consumption deviation.
        """
        range_pwl = range(N_PIECEWISE_LINEAR - 1)
        
        # Piecewise linear (PWL) prosumer consumption
        # Represent prosumer consumption as a sum of multiple pieces from PWL
        self.model.addConstrs(self.m_var['consumption'][i,t] == prosumer_spec['min_consumption'][i][t] 
                              + gp.quicksum(self.m_var['pwl_consumption'][s,i,t] for s in range_pwl) 
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        # Define the upper bound for each PWL consumption
        self.model.addConstrs(self.m_var['pwl_consumption'][s,i,t] <= piecewise_dict['step_range'][i][t]
                              for s in range_pwl for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
        # Rebound effect constraint
        self.model.addConstrs(gp.quicksum(self.m_var['consumption'][i,t] for t in REBOUND_HORIZON) == 
                              gp.quicksum(prosumer_spec['df_consumption'][i][t] for t in REBOUND_HORIZON) 
                              + previous_consumption_deviation[i] for i in self.prosumer_list)
        
    def _set_net_energy_constraint(self, prosumer_spec):
        """
        Define net energy constraints.
        """
        big_M = 1000

        # Prosumers net energy definition
        self.model.addConstrs(self.m_var['consumption'][i,t] - self.m_var['used_solar'][i,t] == 
                              self.m_var['net_positive'][i,t] - self.m_var['export_energy'][i,t]
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
        # Limit the amount of solar energy used in each interval
        self.model.addConstrs(self.m_var['used_solar'][i,t] <= prosumer_spec['df_solar'][i][t] 
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
        # Complementarity constraints for positive and export energy (negative net energy)
        self.model.addConstrs(self.m_var['net_positive'][i,t] <= big_M * (1 - self.m_var['net_binary'][i,t]) 
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        self.model.addConstrs(self.m_var['export_energy'][i,t] <= big_M * self.m_var['net_binary'][i,t]
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
    def _set_export_limit_constraint(self, prosumer_spec):
        # Export limit during solar soak periods
        self.model.addConstrs(self.m_var['export_energy'][i,t] <= prosumer_spec['export_limit'][i][t]
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
    def _set_solar_credit_constraint(self, initial_cumulative_solar_credit):

        # Define actual consumption from the grid for billing purpose
        self.model.addConstrs(self.m_var['net_positive'][i,t] - self.m_var['credit_usage'][i,t] == 
                              self.m_var['import_energy'][i,t] 
                              for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
        # Define cummulative solar credit
        self.model.addConstrs(self.m_var['cumulative_solar_credit'][i,t] == initial_cumulative_solar_credit[i] +
                              gp.quicksum(self.m_var['export_energy'][i,j] - self.m_var['credit_usage'][i,j]
                                       for j in range(t+1)) for i in self.prosumer_list for t in OPTIMISATION_HORIZON)

    def _update_consumption_deviation(self, prosumer_spec, previous_consumption_deviation):
    
        # Get consumption deviation from expected consumption in the first interval of the rolling horizon
        first_interval = 0
        self.model.addConstrs(self.m_var['consumption_deviation'][i] == prosumer_spec['df_consumption'][i][first_interval] 
                              + previous_consumption_deviation[i] - self.m_var['consumption'][i,first_interval] 
                              for i in self.prosumer_list)
        
    def _set_objective_function(self, prosumer_spec, piecewise_dict, spot_price):
    
        range_pwl = range(N_PIECEWISE_LINEAR - 1)
        
        # (Dis)comfort value from the prosumer's consumption
        comfort_value = gp.quicksum(gp.quicksum(piecewise_dict['slope'][i][t][s]*self.m_var['pwl_consumption'][s,i,t] for s in range_pwl)
                                    for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        # Market factor represents non-energy charges from the wholesale market
        # Assume it as a factor of the spot price
        energy_cost = gp.quicksum(spot_price[t] * self.m_var['import_energy'][i,t] * MARKET_COST_FACTOR
                                    for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        # Network usage cost
        network_usage_cost = gp.quicksum(self.m_var['net_positive'][i,t]*prosumer_spec['import_charges'][i][t]
                                         + self.m_var['export_energy'][i,t]*prosumer_spec['export_charges'][i][t]
                                         for i in self.prosumer_list for t in OPTIMISATION_HORIZON)
        
        # Give incentive to export energy to the grid instead of spilling solar energy
        # Ensure minimal effect with a small constant
        solar_storing_incentive = gp.quicksum(self.m_var['cumulative_solar_credit'][i,t]*0.000001 for i in self.prosumer_list 
                                for t in OPTIMISATION_HORIZON)
        
        self.model.setObjective(comfort_value - energy_cost - network_usage_cost + solar_storing_incentive, gp.GRB.MAXIMIZE)

    def optimise_model(self, rolling_th, previous_consumption_deviation, initial_cumulative_solar_credit):
        """
        Optimize the rebound effect for the given rolling horizon.

        Args:
            rolling_th (int): The current rolling horizon.
            previous_consumption_deviation (list): A list of previous consumption deviation for all prosumers.
            initial_cumulative_solar_credit (list): A list of initial cumulative credit usage for all prosumers.
        """

        ### Get constant values
        prosumer_spec = self.parameter_handler.get_prosumer_spec(rolling_th)
        spot_price, reference_price = self.parameter_handler.get_price_spec(rolling_th)
        piecewise_dict = self.parameter_handler.prosumer_comfort_piecewise(prosumer_spec, reference_price)
        
        if previous_consumption_deviation is None:
            previous_consumption_deviation = np.zeros(len(self.prosumer_list))
        if initial_cumulative_solar_credit is None:
            initial_cumulative_solar_credit = np.zeros(len(self.prosumer_list))
        
        ### Model declaration
        self.model = gp.Model('ProsumerRHO')
        self.model.setParam('OutputFlag', 0)
        
        ## Decision variables
        self._set_model_var()
        ## Constraints
        self._set_consumption_constraint(prosumer_spec, piecewise_dict, previous_consumption_deviation)
        self._set_net_energy_constraint(prosumer_spec)
        self._set_export_limit_constraint(prosumer_spec)
        self._set_solar_credit_constraint(initial_cumulative_solar_credit)
        # Axiliary constraint to update consumption deviation over receding horizons
        self._update_consumption_deviation(prosumer_spec, previous_consumption_deviation)
        ## Objective function
        self._set_objective_function(prosumer_spec, piecewise_dict, spot_price)
        
        ## Run the optimization model
        self.model.optimize()

        return self.model

    def _extract_optimised_data(self):
        """
        Extract and process the data from the optimisation model.
        """
        data_dict = {}
        
        # Define variable names and corresponding dictionaries
        var_names = ['consumption', 'credit_usage', 'cumulative_solar_credit', 
                     'net_positive', 'import_energy', 'export_energy', 'used_solar']
        dict_names = [var_name + '_dict' for var_name in var_names]
        
        # Create a dictionary for each type of data
        for j in range(len(var_names)):
            data_dict[dict_names[j]] = {}
            for i in self.prosumer_list:
                data = [self.model.getVarByName('{}[{},{}]'.format(var_names[j], i, t)).x for t in OPTIMISATION_HORIZON]
                data_dict[dict_names[j]][f'prosumer_{str(i)}'] = data
        
        return data_dict
    
    def compile_optimised_data(self, rolling_th):
        """
        Generate and process the data from the optimization model.

        Args:
            rolling_th (int): The current rolling horizon.
        """
        
        data_dict = self._extract_optimised_data()

        # Create pandas dataframes and store them in a dictionary
        dataframes_dict = {
            key: pd.DataFrame(value)
            for key, value in data_dict.items()
        }

        # Add timeline for these rolling dataframes
        time_range = self.parameter_handler.df_prosumer.iloc[rolling_th: rolling_th + len(OPTIMISATION_HORIZON)]["time"].values
        for df_ in dataframes_dict.values():
            df_['time'] = time_range
        
        # Melt the process dataframes like the original df
        for key, df_ in dataframes_dict.items():
            df_ = df_.melt(id_vars=['time'], value_vars=[f'prosumer_{str(i)}' for i in self.prosumer_list], 
                           var_name='dataid', value_name=key.split('_dict')[0])
            dataframes_dict[key] = df_

        # Merge the process dataframes
        initial_key = list(dataframes_dict.keys())[0]
        df_optimised_data = dataframes_dict[initial_key].copy()
        for key in list(dataframes_dict.keys())[1:]:
            df_optimised_data = df_optimised_data.merge(dataframes_dict[key], on=['time', 'dataid'])

        # Merge with solar
        df_solar_temp = self.parameter_handler.df_prosumer[['time', 'solar']].copy()
        df_solar_temp = df_solar_temp.reset_index()
        temp_dataid = df_solar_temp['dataid'].values
        temp_dataid = [f'prosumer_{str(i)}' for i in temp_dataid]
        df_solar_temp['dataid'] = temp_dataid
        
        df_optimised_data = df_optimised_data.merge(df_solar_temp, on=['time', 'dataid'])
        
        return df_optimised_data

class ProsumerRho:

    def __init__(self, prosumer_model):
        self.prosumer_model = prosumer_model
        self.prosumer_list = prosumer_model.prosumer_list
        self.dfs_rolling_data = None
        self.dfs_agg_rolling = None
    
    def _get_spot_price(self, rolling_th):
        """
        Get the spot price and the reference price in the current rolling horizon.
        """
        spot_price,_ = self.prosumer_model.parameter_handler.get_price_spec(rolling_th)
        return spot_price

    def rolling_optimisation(self, n_receding_horizons):
        """
        Run the optimization model for multiple receding horizons.

        Args:
            n_receding_horizons (int): The number of receding horizons.

        Returns:
            list: Processed data for each receding horizon.
        """
        self.dfs_rolling_data = []
        for interval in range(n_receding_horizons):
            print('INTERVAL:', interval)
            
            if interval == 0:
                model = self.prosumer_model.optimise_model(rolling_th=interval, previous_consumption_deviation=None, 
                                                           initial_cumulative_solar_credit=None)
            else:
                model = self.prosumer_model.optimise_model(rolling_th=interval, previous_consumption_deviation=prosumers_consumption_deviation, 
                                                           initial_cumulative_solar_credit=prosumers_cumulative_solar_credit)
                    
            prosumers_consumption_deviation = [v.x for v in model.getVars() if 'consumption_deviation' in v.varName]
            prosumers_cumulative_solar_credit = [model.getVarByName('cumulative_solar_credit[{},0]'.format(i)).x 
                                                 for i in self.prosumer_list]
            
            df_optimised_data = self.prosumer_model.compile_optimised_data(rolling_th=interval)

            spot_price = self._get_spot_price(rolling_th=interval)
            df_optimised_data['pd_price'] = np.tile(spot_price.values, len(self.prosumer_list))

            self.dfs_rolling_data.append(df_optimised_data)
            
        return self.dfs_rolling_data
    
    # Save the optimised data to csv files
    def save_rolling_data(self, folder_name):
        if self.dfs_rolling_data is None:
            raise ValueError('No data to save. Please run the rolling optimisation first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for interval, df_optimised_data in enumerate(self.dfs_rolling_data):
            # Save without index
            df_optimised_data.to_csv(f'{folder_name}/interval_{interval}_pd_price.csv', index=False)

    def compile_aggregate_data(self, rolling_data_folder=None):
        if self.dfs_rolling_data is None or rolling_data_folder is not None:
            def read_csv_file(csv_file):
                df = pd.read_csv(csv_file, low_memory=False, float_precision='round_trip', index_col='dataid')
                df['time'] = pd.to_datetime(df['time'])
                # Drop nan values if any
                df = df.dropna()
                return df
            self.dfs_rolling_data = [read_csv_file(f'{rolling_data_folder}/interval_{interval}_pd_price.csv') 
                                     for interval in range(len((glob.glob(f'{rolling_data_folder}/*.csv'))))]

        group_sum = ['consumption', 'solar', 'credit_usage', 'cumulative_solar_credit', 
                     'net_positive', 'import_energy', 'export_energy', 'used_solar']
        group_mean = ['pd_price']
        agg_dict = {s: 'sum' for s in group_sum}
        agg_dict.update({m: 'mean' for m in group_mean})
        self.dfs_agg_rolling = [df_optimised_data.groupby('time').agg(agg_dict) for df_optimised_data in self.dfs_rolling_data]
        return self.dfs_agg_rolling
    
    def save_aggregate_rolling_data(self, folder_name):
        if self.dfs_agg_rolling is None:
            raise ValueError('No data to save. Please compile the aggregate rolling optimised data first.')
        # If folder does not exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for interval, df_agg_optimised in enumerate(self.dfs_agg_rolling):
            df_agg_optimised.to_csv(f'{folder_name}/aggregate_interval_{interval}_pd_price.csv')