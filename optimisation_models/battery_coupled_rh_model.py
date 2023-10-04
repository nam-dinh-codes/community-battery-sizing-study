import pandas as pd
import numpy as np
import os
import gurobipy as gp
import itertools
from .battery_optimisation import BatteryModel

MARKET_COST_FACTOR = 1.2
NUM_INTERVALS_DAY = 48

class BatteryCoupledRhModel(BatteryModel):
    def __init__(self, lookahead_horizon, receding_horizon, optimisation_period):
        """
        Initialise the BatteryCoupledRhModel class.
        
        Parameters:
            lookahead_horizon: Time intervals for the lookahead horizon.
            receding_horizon: Index for the receding horizon.
            optimisation_period: Index for the optimisation period. Or the scenarios (months)
        """
        self.lookahead_horizon = lookahead_horizon
        self.receding_horizon = receding_horizon
        # Instead of coupling too many receding horizons together, optimisation period (or scenarios)
        # are introduced to avoid one receding horizon from looking too far ahead into the future.
        self.optimisation_period = optimisation_period
        # Generate list of tuples for Gurobi variables declaration
        self.optimisation_horizon = list(itertools.product(optimisation_period, receding_horizon, lookahead_horizon))
        super().__init__(self.optimisation_horizon)

    def _set_battery_soc_evolution(self, initial_soc, battery_spec):
        """Modify the SoC evolution from BatteryModel because receding horizons are coupled together."""
        # Update the SoC for every interval within the lookahead horizon
        self.model.addConstrs(self.m_var['battery_energy'][m,j,h] == self.m_var['battery_energy'][m,j,h-1] 
                              + self.m_var['charging_energy_positive'][m,j,h] 
                              - self.m_var['charging_energy_negative'][m,j,h] / battery_spec['efficiency'] 
                              for m in self.optimisation_period for j in self.receding_horizon for h in self.lookahead_horizon[1:])
        self.model.addConstrs(self.m_var['battery_energy'][m,0,0] == initial_soc + self.m_var['charging_energy_positive'][m,0,0] 
                              - self.m_var['charging_energy_negative'][m,0,0] / battery_spec['efficiency'] for m in self.optimisation_period)
        # Update the SoC at beginning of the lookahead horizon based on the committed SoC in previous receding horizon
        self.model.addConstrs(self.m_var['battery_energy'][m,j,0] == self.m_var['battery_energy'][m,j-1,0] 
                              + self.m_var['charging_energy_positive'][m,j,0] 
                              - self.m_var['charging_energy_negative'][m,j,0] / battery_spec['efficiency'] 
                              for m in self.optimisation_period for j in self.receding_horizon[1:])
        # Set ending SoC of the lookahead horizon to be the same as the beginning SoC (committed SoC in previous receding horizon)
        self.model.addConstrs(self.m_var['battery_energy'][m,j,len(self.lookahead_horizon) - 1] == self.m_var['battery_energy'][m,j-1,0] 
                              for m in self.optimisation_period for j in self.receding_horizon[1:])
        self.model.addConstrs(self.m_var['battery_energy'][m,0,len(self.lookahead_horizon) - 1] == initial_soc for m in self.optimisation_period)
        # Ensure that the SoC equals the initial SoC at the end of the sizing horizon for fair comparison
        self.model.addConstrs(self.m_var['battery_energy'][m,len(self.receding_horizon) - 1,0] == initial_soc for m in self.optimisation_period)

    def _set_net_energy_constraint(self, prosumer_spec):
        """Modify the net energy constraint from BatteryModel because receding horizons are coupled together."""
        big_M = 1000    # Big M for complementarity constraint
        # prosumer_spec is indexed differently from thje BatteryModel class
        self.model.addConstrs(prosumer_spec['net_positive'][(m*len(self.receding_horizon) + j)*len(self.lookahead_horizon) + h] 
                              - prosumer_spec['export_energy'][(m*len(self.receding_horizon) + j)*len(self.lookahead_horizon) + h]
                              + self.m_var['charging_energy'][m,j,h] == self.m_var['community_net_positive'][m,j,h]
                              - self.m_var['community_net_negative'][m,j,h] 
                              for m in self.optimisation_period for j in self.receding_horizon for h in self.lookahead_horizon)
        # Complementarity constraint for community net energy
        self.model.addConstrs(self.m_var['community_net_positive'][t] <= big_M * (1 - self.m_var['community_net_binary'][t])
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['community_net_negative'][t] <= big_M * self.m_var['community_net_binary'][t]
                              for t in self.optimisation_horizon)
        # Constraint for defining energy import from the grid to charge the battery
        self.model.addConstrs(self.m_var['charging_energy_positive'][m,j,h] 
                              - prosumer_spec['export_energy'][(m*len(self.receding_horizon) + j)*len(self.lookahead_horizon) + h] 
                              <= self.m_var['grid_charging_positive'][m,j,h]
                              for m in self.optimisation_period for j in self.receding_horizon for h in self.lookahead_horizon)

    def _set_objective_function(self, fee_information, battery_spec):
        """Modify the objective function from BatteryModel because receding horizons are coupled together."""
        # Calculate different components of the objective function
        energy_cost = gp.quicksum((fee_information['spot_price'][(m*len(self.receding_horizon) + j)*len(self.lookahead_horizon) + h] 
                                   * self.m_var['community_net_positive'][m,j,h] * MARKET_COST_FACTOR)
                                  for m in self.optimisation_period for j in self.receding_horizon for h in self.lookahead_horizon)
        dnsp_cost = gp.quicksum(fee_information['LUoS'] * self.m_var['grid_charging_positive'][t]
                                for t in self.optimisation_horizon)
        battery_opex = gp.quicksum(fee_information['charging_fee'] * self.m_var['charging_energy_negative'][t]
                                   for t in self.optimisation_horizon)
        
        n_optimised_days = len(self.optimisation_horizon) / (NUM_INTERVALS_DAY * len(self.lookahead_horizon))
        peak_demand_cost = fee_information['peak_charge'] * self.m_var['max_net_energy'] * n_optimised_days / battery_spec['time_resolution']
        # Get battery cost based on 10 years warranty period
        battery_cost = battery_spec['price'] * self.m_var['battery_capacity'] * n_optimised_days / (365 * battery_spec['warranty_period'])

        # Set the objective function to minimise the total cost
        lookahead_weighted_cost = (energy_cost + dnsp_cost + battery_opex) / len(self.lookahead_horizon)
        self.model.setObjective(lookahead_weighted_cost  + peak_demand_cost + battery_cost, gp.GRB.MINIMIZE)

class BatteryCoupledRhSizing():
    def __init__(self, parameter_handler):
        """
        Initialises the BatteryCoupledRhSizing class.
        
        Parameters:
            parameter_handler (object): Instance of OptimisationParameters from battery_optimisation file.
        """
        self.parameter_handler = parameter_handler
        self.lookahead_horizon = np.arange(32)
        self.receding_horizon = np.arange(48*7)
        self.monthly_list = np.arange(0, 12)
        self.battery_model = BatteryCoupledRhModel(self.lookahead_horizon, self.receding_horizon, self.monthly_list)
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

    def sizing_optimisation(self, verbose=1):
        """Perform sizing optimisation.

        Parameters:
            verbose (int): Verbose level for the Gurobi solver. Default is 1.
        
        Returns:
            pd.DataFrame: DataFrame containing optimised battery operation data.
        """
        vars_to_get = ['battery_energy', 'charging_energy', 'charging_energy_positive', 
                       'charging_energy_negative', 'community_net_positive', 
                       'community_net_negative', 'grid_charging_positive']
        batt_oper_dict = {var: [] for var in vars_to_get}

        # Get constant values
        df_aggregate, battery_spec, prosumer_spec, fee_information = self._get_optimisation_parameter()
        # Run the optimisation model
        model = self.battery_model.optimise_model(battery_spec, prosumer_spec, fee_information, 
                                                  initial_soc=0, current_max_net_energy=0, verbose=verbose)

        # Fetch the optimisation results
        for var in vars_to_get:
            batt_oper_dict[var] = [model.getVarByName(f'{var}[{month},{j},0]').x 
                                   for month in self.monthly_list for j in self.receding_horizon]
        batt_oper_dict['capacity'] = model.getVarByName('battery_capacity').x
        batt_oper_dict['time'] = df_aggregate.reset_index().groupby('rolling_th').first()['time'].values

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
        self.df_battery_optimised.to_csv(f'{folder_name}/battery_{capacity}_coupled_rh_sizing.csv')