import gurobipy as gp

MARKET_COST_FACTOR = 1.2

class OptimisationParameters:
    def __init__(self, df_prosumer_agg, battery_capacity=None):
        self.df_prosumer_agg = df_prosumer_agg
        self.battery_capacity = battery_capacity

    def get_battery_spec(self):
        """
        Battery specs
        Output a dictionary
        """

        battery_spec = dict()
        battery_spec['lower_SOC'] = 0
        battery_spec['upper_SOC'] = 1.0
        battery_spec['price'] = 800 # $/kWh capacity
        battery_spec['duration'] = 2 # [h]
        battery_spec['efficiency'] = 0.9 # Round-trip efficiency
        battery_spec['time_resolution'] = 0.5 # [h]
        battery_spec['warranty_period'] = 10 # [years]
        # Battery capacity can be set to None if it is not originally defined
        battery_spec['capacity'] = self.battery_capacity # [kWh]
        
        return battery_spec
    
    def get_prosumer_agg_data(self, rolling_th):
        if rolling_th == 'full':
            return self.df_prosumer_agg
        return self.df_prosumer_agg[self.df_prosumer_agg['rolling_th'] == rolling_th]

    def get_prosumer_spec(self, df_aggregate):
        """
        Prosumer specs
        Output a dictionary
        """
        columns_of_interest = ['net_positive', 'export_energy', 'import_energy']

        prosumer_spec = {}
        for column in columns_of_interest:
            prosumer_spec[column] = df_aggregate[column]

        return prosumer_spec
    
    def get_price_spec(self, df_aggregate):
        """
        Get the list of spot price and the maximum price in the current rolling horizon.
        """
        fee_information = dict()
        fee_information['spot_price'] = df_aggregate['price']
        fee_information['charging_fee'] = 0.032 # $/kWh
        fee_information['LUoS'] = 0.016098 # $/kWh
        fee_information['peak_charge'] = 0.331069 # $/kW/day
        
        return fee_information
    

class BatteryModel():
    def __init__(self, optimisation_horizon):
        self.optimisation_horizon = optimisation_horizon
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
        add_model_vars(self.optimisation_horizon, name='charging_energy', lb=-gp.GRB.INFINITY)
        add_model_vars(self.optimisation_horizon, name='battery_energy')
        add_model_vars(self.optimisation_horizon, name='charging_energy_positive')
        add_model_vars(self.optimisation_horizon, name='charging_energy_negative')

        # Separate local net
        add_model_vars(self.optimisation_horizon, name='community_net_positive')
        add_model_vars(self.optimisation_horizon, name='community_net_negative')
        add_model_vars(self.optimisation_horizon, name='community_net_binary', vtype=gp.GRB.BINARY)

        # Separate energy charged from utility grid
        add_model_vars(self.optimisation_horizon, name='grid_charging_positive')

        # Maximum net energy
        self.m_var['max_net_energy'] = self.model.addVar(name='max_net_energy')

        # Battery capacity
        self.m_var['battery_capacity'] = self.model.addVar(name='battery_capacity')
        
    def _set_battery_boundaries(self, battery_spec):
        """
        Set battery boundaries
        """
        # Charging constraints
        charging_power = self.m_var['battery_capacity'] / battery_spec['duration']
        # Multiple by time resolution (0.5) representing half an hour
        self.model.addConstrs(self.m_var['charging_energy'][t] <= battery_spec['time_resolution'] * charging_power
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['charging_energy'][t] >= -battery_spec['time_resolution'] * charging_power
                              for t in self.optimisation_horizon)
        # Complemetarity constraint for battery charging
        self.model.addConstrs(self.m_var['charging_energy'][t] == self.m_var['charging_energy_positive'][t] 
                              - self.m_var['charging_energy_negative'][t] for t in self.optimisation_horizon)
        
        # Define bounbdaries for SOC
        self.model.addConstrs(self.m_var['battery_energy'][t] >= battery_spec['lower_SOC'] * self.m_var['battery_capacity']
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['battery_energy'][t] <= battery_spec['upper_SOC'] * self.m_var['battery_capacity']
                              for t in self.optimisation_horizon)
        
    def _set_battery_soc_evolution(self, initial_SOC, battery_spec):
        
        # SOC determination
        self.model.addConstrs(self.m_var['battery_energy'][t] == initial_SOC + 
                              gp.quicksum(self.m_var['charging_energy_positive'][j] 
                                          - self.m_var['charging_energy_negative'][j] / battery_spec['efficiency'] 
                                          for j in range(t + 1)) for t in self.optimisation_horizon)

        # Ending SOC of the receding horizon is equal the initial SOC
        self.model.addConstr(self.m_var['battery_energy'][len(self.optimisation_horizon) - 1] == initial_SOC)
    
    def _set_net_energy_constraint(self, prosumer_spec):
        big_M = 1000

        self.model.addConstrs(prosumer_spec['net_positive'][t] - prosumer_spec['export_energy'][t]
                              + self.m_var['charging_energy'][t] == self.m_var['community_net_positive'][t] 
                              - self.m_var['community_net_negative'][t] for t in self.optimisation_horizon)

        # Complementarity constraint for community net energy
        self.model.addConstrs(self.m_var['community_net_positive'][t] <= big_M * (1 - self.m_var['community_net_binary'][t])
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['community_net_negative'][t] <= big_M * self.m_var['community_net_binary'][t]
                              for t in self.optimisation_horizon)

        self.model.addConstrs(self.m_var['charging_energy_positive'][t] - prosumer_spec['export_energy'][t] <=
                              self.m_var['grid_charging_positive'][t] for t in self.optimisation_horizon)
    
    def _set_peak_demand_constraint(self, current_max_net_energy=0):
        # Define maximum net energy
        self.model.addConstrs(self.m_var['max_net_energy'] >= self.m_var['community_net_positive'][t] 
                              for t in self.optimisation_horizon)
        self.model.addConstr(self.m_var['max_net_energy'] >= current_max_net_energy)

    def _set_battery_capacity(self, battery_spec):
        """
        Set battery capacity if it is not None
        """
        if battery_spec['capacity'] is not None:
            self.model.addConstr(self.m_var['battery_capacity'] == battery_spec['capacity'])
        
    def _set_objective_function(self, fee_information, battery_spec):
        
        energy_cost = gp.quicksum(fee_information['spot_price'][t] * self.m_var['community_net_positive'][t] * MARKET_COST_FACTOR
                                  for t in self.optimisation_horizon)
        dnsp_cost = gp.quicksum(fee_information['LUoS'] * self.m_var['grid_charging_positive'][t]
                                for t in self.optimisation_horizon)
        battery_opex = gp.quicksum(fee_information['charging_fee'] * self.m_var['charging_energy_negative'][t]
                                   for t in self.optimisation_horizon)
        
        peak_demand_cost = fee_information['peak_charge'] * 365 * self.m_var['max_net_energy'] / battery_spec['time_resolution']

        # Set the objective function to maximize the total profit
        self.model.setObjective(energy_cost + dnsp_cost + battery_opex + peak_demand_cost, gp.GRB.MINIMIZE)
    
    def optimise_model(self, battery_spec, prosumer_spec, fee_information, 
                       initial_SOC, current_max_net_energy, verbose=0):
        
        ### Model declaration
        self.model = gp.Model('BatteryModel')
        self.model.setParam('OutputFlag', verbose)
        self.model.setParam('TimeLimit', 700)
        
        ## Decision variables
        self._set_model_var()
        ## Constraints
        self._set_battery_boundaries(battery_spec)
        self._set_battery_soc_evolution(initial_SOC, battery_spec)
        self._set_net_energy_constraint(prosumer_spec)
        self._set_peak_demand_constraint(current_max_net_energy)
        self._set_battery_capacity(battery_spec)
        ## Objective function
        self._set_objective_function(fee_information, battery_spec)

        ## Run the optimization model
        self.model.optimize()

        return self.model