import gurobipy as gp

MARKET_COST_FACTOR = 1.2

class OptimisationParameters:
    """
    Class to hold the parameters required for battery optimisation.
    """

    def __init__(self, df_prosumer_agg, battery_capacity=None):
        """
        Initialise the OptimisationParameters object with prosumer data and battery capacity.
        
        Parameters:
        - df_prosumer_agg (DataFrame): Aggregated data for prosumer.
        - battery_capacity (float): Capacity of the battery (Optional).
        """
        self.df_prosumer_agg = df_prosumer_agg
        self.battery_capacity = battery_capacity

    def get_battery_spec(self):
        """
        Get the specifications for the battery.
        
        Returns:
        - dict: Dictionary containing battery specifications like lower and upper SoC, price, etc.
        """

        # Initialize battery specification dictionary
        battery_spec = {
            'lower_SOC': 0,
            'upper_SOC': 1.0,
            'price': 800,  # $/kWh capacity
            'duration': 2,  # hours
            'efficiency': 0.9,  # Round-trip efficiency
            'time_resolution': 0.5,  # hours
            'warranty_period': 10,  # years
            'capacity': self.battery_capacity  # kWh
        }
        return battery_spec
    
    def get_prosumer_agg_data(self, rolling_th):
        if rolling_th == 'full':
            return self.df_prosumer_agg
        return self.df_prosumer_agg[self.df_prosumer_agg['rolling_th'] == rolling_th]

    def get_prosumer_spec(self, df_aggregate):
        """
        Extract prosumer aggregated data.

        Parameters:
            df_aggregate (DataFrame): DataFrame containing prosumer data.

        Returns:
            dict: Dictionary containing relevant prosumer specifications.
        """
        columns_of_interest = ['net_positive', 'export_energy', 'import_energy']
        return {column: df_aggregate[column] for column in columns_of_interest}
    
    def get_price_spec(self, df_aggregate):
        """
        Get pricing specifications.

        Parameters:
            df_aggregate (DataFrame): DataFrame containing (forecast) spot prices.

        Returns:
            dict: Dictionary containing relevant pricing specifications.
        """
        fee_information = {
            'spot_price': df_aggregate['price'],
            'charging_fee': 0.032,  # $/kWh
            'LUoS': 0.016098,  # $/kWh
            'peak_charge': 0.331069  # $/kW/day
        }
        return fee_information
    

class BatteryModel():
    """
    Class to perform battery optimisation using Gurobi.

    Attributes:
        optimisation_horizon (list): List of time periods or list of tuples for optimisation.
        model (Gurobi Model): Gurobi optimization model.
        m_var (dict): Dictionary of decision variables in the model.
    """
    def __init__(self, optimisation_horizon):
        """
        Initialise the BatteryModel object.

        Parameters:
            optimisation_horizon (list): List of time periods or list of tuples for optimisation.
        """
        self.optimisation_horizon = optimisation_horizon
        self.model = None
        self.m_var = None
    
    def _set_model_var(self):
        """
        Add decision variables for the model.
        """
        def add_model_vars(*args, **kwargs):
            """Helper function to add variables to the model and store them in m_var."""
            var_name = kwargs['name']
            self.m_var[var_name] = self.model.addVars(*args, **kwargs)

        # Create a dict of model variables
        self.m_var = dict()

        # Add battery-related variables
        add_model_vars(self.optimisation_horizon, name='charging_energy', lb=-gp.GRB.INFINITY)
        add_model_vars(self.optimisation_horizon, name='battery_energy')
        add_model_vars(self.optimisation_horizon, name='charging_energy_positive')
        add_model_vars(self.optimisation_horizon, name='charging_energy_negative')
        self.m_var['battery_capacity'] = self.model.addVar(name='battery_capacity')
        # Add community-related variables
        add_model_vars(self.optimisation_horizon, name='community_net_positive')
        add_model_vars(self.optimisation_horizon, name='community_net_negative')
        add_model_vars(self.optimisation_horizon, name='community_net_binary', vtype=gp.GRB.BINARY)
        # Add grid-related variables
        add_model_vars(self.optimisation_horizon, name='grid_charging_positive')
        # Add variables for maximum net energy
        self.m_var['max_net_energy'] = self.model.addVar(name='max_net_energy')
        
    def _set_battery_boundaries(self, battery_spec):
        """
        Add battery boundaries constraitns for the model.

        Parameters:
            battery_spec (dict): Dictionary of battery specifications.
        """
        # Calculate the maximum charging power based on battery capacity and duration
        charging_power = self.m_var['battery_capacity'] / battery_spec['duration']
        # Set charging constraints based on calculated charging power
        self.model.addConstrs(self.m_var['charging_energy'][t] <= battery_spec['time_resolution'] * charging_power
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['charging_energy'][t] >= -battery_spec['time_resolution'] * charging_power
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['charging_energy'][t] == self.m_var['charging_energy_positive'][t] 
                              - self.m_var['charging_energy_negative'][t] for t in self.optimisation_horizon)
        # Constraint for defining bounbdaries for SoC
        self.model.addConstrs(self.m_var['battery_energy'][t] >= battery_spec['lower_SOC'] * self.m_var['battery_capacity']
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['battery_energy'][t] <= battery_spec['upper_SOC'] * self.m_var['battery_capacity']
                              for t in self.optimisation_horizon)
        
    def _set_battery_soc_evolution(self, initial_soc, battery_spec):
        """
        Add battery SOC evolution constraints for the model.

        Parameters:
            initial_soc (float): Initial SoC before the start of the optimisation.
            battery_spec (dict): Dictionary of battery specifications.
        """
        # SOC determination
        self.model.addConstrs(self.m_var['battery_energy'][t] == initial_soc + 
                              gp.quicksum(self.m_var['charging_energy_positive'][j] 
                                          - self.m_var['charging_energy_negative'][j] / battery_spec['efficiency'] 
                                          for j in range(t + 1)) for t in self.optimisation_horizon)

        # Ending SOC of the receding horizon equals the initial SOC
        self.model.addConstr(self.m_var['battery_energy'][len(self.optimisation_horizon) - 1] == initial_soc)
    
    def _set_net_energy_constraint(self, prosumer_spec):
        """
        Add net energy constraint for the model.

        Parameters:
            prosumer_spec (dict): Dictionary containing prosumer specifications.
        """
        big_M = 1000    # Big M for complementarity constraint

        self.model.addConstrs(prosumer_spec['net_positive'][t] - prosumer_spec['export_energy'][t]
                              + self.m_var['charging_energy'][t] == self.m_var['community_net_positive'][t] 
                              - self.m_var['community_net_negative'][t] for t in self.optimisation_horizon)

        # Complementarity constraint for community net energy
        self.model.addConstrs(self.m_var['community_net_positive'][t] <= big_M * (1 - self.m_var['community_net_binary'][t])
                              for t in self.optimisation_horizon)
        self.model.addConstrs(self.m_var['community_net_negative'][t] <= big_M * self.m_var['community_net_binary'][t]
                              for t in self.optimisation_horizon)
        # Constraint for defining energy import from the grid to charge the battery
        self.model.addConstrs(self.m_var['charging_energy_positive'][t] - prosumer_spec['export_energy'][t] <=
                              self.m_var['grid_charging_positive'][t] for t in self.optimisation_horizon)
    
    def _set_peak_demand_constraint(self, current_max_net_energy=0):
        """
        Set the peak demand constraint for the model.

        Parameters:
            current_max_net_energy (float): The current maximum net energy. Defaults to 0.
        """
        # Define max net demand based on potential maximum net demand in the look-ahead horizon
        self.model.addConstrs(self.m_var['max_net_energy'] >= self.m_var['community_net_positive'][t] 
                              for t in self.optimisation_horizon)
        # Define max net demand based on observed peak demand in previous receding horizons
        self.model.addConstr(self.m_var['max_net_energy'] >= current_max_net_energy)

    def _set_battery_capacity(self, battery_spec):
        """
        Set battery capacity if it is not None
        """
        if battery_spec['capacity'] is not None:
            self.model.addConstr(self.m_var['battery_capacity'] == battery_spec['capacity'])
        
    def _set_objective_function(self, fee_information, battery_spec):
        """
        Set the objective function for the optimisation model.

        Parameters:
            fee_information (dict): Dictionary containing fee information.
            battery_spec (dict): Dictionary containing battery specifications.
        """
        # Calculate different components of the objective function
        energy_cost = gp.quicksum(fee_information['spot_price'][t] * self.m_var['community_net_positive'][t] * MARKET_COST_FACTOR
                                  for t in self.optimisation_horizon)
        dnsp_cost = gp.quicksum(fee_information['LUoS'] * self.m_var['grid_charging_positive'][t]
                                for t in self.optimisation_horizon)
        battery_opex = gp.quicksum(fee_information['charging_fee'] * self.m_var['charging_energy_negative'][t]
                                   for t in self.optimisation_horizon)
        
        peak_demand_cost = fee_information['peak_charge'] * 365 * self.m_var['max_net_energy'] / battery_spec['time_resolution']

        # Set the objective function to minimise the total cost
        self.model.setObjective(energy_cost + dnsp_cost + battery_opex + peak_demand_cost, gp.GRB.MINIMIZE)
    
    def optimise_model(self, battery_spec, prosumer_spec, fee_information, 
                       initial_soc, current_max_net_energy, verbose=0):
        """
        Run the optimisation model with the given parameters.

        Parameters:
            battery_spec (dict): Battery specifications.
            prosumer_spec (dict): Prosumer specifications.
            fee_information (dict): Fee information.
            initial_soc (float): Initial SoC.
            current_max_net_energy (float): Observed maximum net energy.
            verbose (int): Verbose level for the Gurobi solver. Default is 0.

        Returns:
            model (Gurobi model): The optimised Gurobi model.
        """
        ### Model declaration
        self.model = gp.Model('BatteryModel')
        self.model.setParam('OutputFlag', verbose)
        # Set the time limit for the optimisation model
        # This is not really needed as the rolling optimisation is solved relatively fast
        self.model.setParam('TimeLimit', 600)
        
        ## Decision variables
        self._set_model_var()
        ## Constraints
        self._set_battery_boundaries(battery_spec)
        self._set_battery_soc_evolution(initial_soc, battery_spec)
        self._set_net_energy_constraint(prosumer_spec)
        self._set_peak_demand_constraint(current_max_net_energy)
        self._set_battery_capacity(battery_spec)
        ## Objective function
        self._set_objective_function(fee_information, battery_spec)

        ## Run the optimization model
        self.model.optimize()

        return self.model