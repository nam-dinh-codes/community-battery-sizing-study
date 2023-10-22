import pandas as pd
import numpy as np

MARKET_COST_FACTOR = 1.2
NUM_INTERVALS_DAY = 48

class ProjectCost():
    def __init__(self, parameter_handler, df_prosumer_agg):
        self.parameter_handler = parameter_handler
        self.df_prosumer_agg = df_prosumer_agg
        self.battery_spec = None
        self.prosumer_spec = None
        self.fee_information = None
        self._get_optimisation_parameter()

    def _get_optimisation_parameter(self):
        """Get optimisation parameters
        
        Returns:
            tuple: Contains aggregate DataFrame, battery, prosumer and fee specifications.
        """
        df_aggregate = self.parameter_handler.get_prosumer_agg_data(rolling_th='full')
        self.battery_spec = self.parameter_handler.get_battery_spec()
        self.prosumer_spec = self.parameter_handler.get_prosumer_spec(df_aggregate)
        self.fee_information = self.parameter_handler.get_price_spec(df_aggregate)
    
    def get_battery_data(self, battery_data_file):
        """Read battery data from csv file.
        
        Returns:
            pd.DataFrame: Dataframe with battery data.
        """
        df_battery_data = pd.read_csv(battery_data_file, low_memory=False, 
                                      float_precision='round_trip', index_col='time', 
                                      parse_dates=True)
        return df_battery_data
    
    def get_project_cost(self, df_battery_data):
        cost_info = {}
        daily_normalise_coeff = NUM_INTERVALS_DAY / df_battery_data.shape[0]

        prosumer_peak = self.df_prosumer_agg['net_positive'].max()
        # Peak reduction in kW
        peak_reduction = (df_battery_data['community_net_positive'].max() - prosumer_peak)  / self.battery_spec['time_resolution']
        # Annualised peak saving
        peak_saving = peak_reduction * self.fee_information['peak_charge'] * 365
        # Annualised energy charge
        energy_charge = np.sum(self.fee_information['spot_price'] * df_battery_data['community_net_positive'] * MARKET_COST_FACTOR) * 365 * daily_normalise_coeff
        # Annualised battery Opex
        battery_opex = np.sum(self.fee_information['charging_fee'] * df_battery_data['charging_energy_negative']) * 365 * daily_normalise_coeff
        # Annualised DNSP charge
        dnsp_charge = np.sum(self.fee_information['LUoS'] * df_battery_data['grid_charging_positive']) * 365 * daily_normalise_coeff
        # Annualised battery cost
        battery_cost = df_battery_data['capacity'].iloc[0] * self.battery_spec['price'] / self.battery_spec['warranty_period']
        # Annualised ground truth cost
        annualised_cost = battery_cost + energy_charge + dnsp_charge + battery_opex + peak_saving
        
        cost_info['ground_truth_cost'] = round(annualised_cost, 2)      # in $/year
        cost_info['battery_cost'] = round(battery_cost, 2)              # in $/year
        cost_info['energy_charge'] = round(energy_charge, 2)            # in $/year
        cost_info['dnsp_charge'] = round(dnsp_charge, 2)                # in $/year
        cost_info['battery_opex'] = round(battery_opex, 2)              # in $/year
        cost_info['peak_saving'] = round(peak_saving, 2)                # in $/year
        cost_info['peak_reduction'] = round(peak_reduction, 2)          # in kW
        return cost_info