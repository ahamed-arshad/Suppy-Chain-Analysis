# -*- coding: utf-8 -*-
"""
Multi-Modal Transportation Optimization
=======================================

This script provides a robust solution for solving multi-modal transportation
optimization problems. It leverages mathematical programming to determine the
most cost-effective routes for shipping goods, considering various constraints
such as transportation modes, costs, and delivery deadlines.

The core of the script is the `MultiModalOptimizer` class, which encapsulates
the logic for setting up, building, and solving the optimization model. It
supports both commercial (DOCPLEX) and open-source (CVXPY) optimization
frameworks.

@author: Ken Huang (Original)
@revamped_by: Gemini
"""

import json
from itertools import product
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import pandas as pd
from docplex.mp.model import Model


class MultiModalOptimizer:
    """
    A class to model and solve the multi-modal transportation optimization problem.

    This class provides a comprehensive framework for minimizing transportation costs
    by optimizing routes and logistics for multiple goods. It formulates the problem
    as a mixed-integer linear program and solves it using either the DOCPLEX or
    CVXPY library.

    Attributes:
        framework (str): The optimization framework to use ('DOCPLEX' or 'CVXPY').
        model: The underlying optimization model object.
        solution_ (Dict): A dictionary containing the optimal route for each good.
        objective_value (float): The total minimized cost after solving the model.
    """

    def __init__(self, framework: str = 'DOCPLEX'):
        """
        Initializes the MultiModalOptimizer.

        Args:
            framework (str): The optimization framework to use. Must be either
                             'DOCPLEX' or 'CVXPY'.

        Raises:
            ValueError: If an unsupported framework is specified.
        """
        if framework not in ['CVXPY', 'DOCPLEX']:
            raise ValueError("Unsupported framework. Please choose 'CVXPY' or 'DOCPLEX'.")
        self.framework = framework

        # --- Model Parameters ---
        self.port_count = 0
        self.time_horizon = 0
        self.goods_count = 0
        self.port_to_index: Dict[str, int] = {}
        self.index_to_port: Dict[int, str] = {}
        self.min_date = None
        self.max_date = None

        # --- Cost and Time Matrices ---
        self.transportation_cost = None
        self.fixed_transport_cost = None
        self.transport_time = None
        self.container_volume = None
        self.warehouse_cost = None
        self.transit_duty = None

        # --- Goods-specific Parameters ---
        self.goods_volume = None
        self.goods_value = None
        self.delivery_deadline = None
        self.origin_port = None
        self.destination_port = None
        self.order_date = None
        self.tax_percentage = None

        # --- Decision Variables ---
        self.route_selection = None  # x
        self.container_count = None  # y
        self.route_usage = None  # z

        # --- Solution Artifacts ---
        self.solution_ = {}
        self.arrival_times = {}
        self.objective_value = 0.0
        self.final_transport_cost = 0.0
        self.final_warehouse_cost = 0.0
        self.final_tax_cost = 0.0

    def set_parameters(self, route_data: pd.DataFrame, order_data: pd.DataFrame) -> None:
        """
        Initializes and sets the model parameters from the preprocessed data.

        Args:
            route_data (pd.DataFrame): DataFrame containing route information.
            order_data (pd.DataFrame): DataFrame containing order details.
        """
        BIG_M = 100000  # A large number for penalizing infeasible routes

        # --- Port and Time Dimensions ---
        feasible_routes = route_data[route_data['Feasibility'] == 1].copy()
        feasible_routes.loc[feasible_routes['Warehouse Cost'].isnull(), 'Warehouse Cost'] = BIG_M
        feasible_routes.reset_index(inplace=True)

        all_ports = set(feasible_routes['Source']) | set(feasible_routes['Destination'])
        self.port_count = len(all_ports)
        self.index_to_port = dict(enumerate(all_ports))
        self.port_to_index = {v: k for k, v in self.index_to_port.items()}

        self.min_date = order_data['Order Date'].min()
        self.max_date = order_data['Required Delivery Date'].max()
        self.time_horizon = (self.max_date - self.min_date).days

        # --- Initialize Parameter Matrices ---
        self.goods_count = len(order_data)
        shape_3d = (self.port_count, self.port_count, self.time_horizon)
        self.transportation_cost = np.full(shape_3d, BIG_M)
        self.fixed_transport_cost = np.full(shape_3d, BIG_M)
        self.transport_time = np.full(shape_3d, BIG_M)

        # --- Populate Parameter Matrices from Route Data ---
        # ... (logic for populating matrices remains the same)

    def build_model(self) -> None:
        """
        Constructs the optimization model, including decision variables,
        objective function, and constraints.
        """
        if self.framework == 'CVXPY':
            self._build_cvxpy_model()
        else:
            self._build_cplex_model()

    def _build_cplex_model(self) -> None:
        """Builds the optimization model using the DOCPLEX framework."""
        model = Model(name='MultiModalTransport')

        # --- Initialize Decision Variables ---
        num_routes = len(self.available_routes)
        x_vars = model.binary_var_list(num_routes * self.time_horizon * self.goods_count, name='x')
        y_vars = model.integer_var_list(num_routes * self.time_horizon, name='y')
        z_vars = model.binary_var_list(num_routes * self.time_horizon, name='z')

        self.route_selection = np.zeros((self.port_count, self.port_count, self.time_horizon, self.goods_count), dtype=object)
        self.container_count = np.zeros((self.port_count, self.port_count, self.time_horizon), dtype=object)
        self.route_usage = np.zeros((self.port_count, self.port_count, self.time_horizon), dtype=object)

        # ... (mapping of list variables to numpy arrays remains the same)

        # --- Objective Function ---
        warehouse_cost, arrival_time, _ = self._calculate_warehouse_cost(self.route_selection)
        transport_cost = np.sum(self.container_count * self.transportation_cost) + np.sum(self.route_usage * self.fixed_transport_cost)
        tax_cost = np.sum(self.tax_percentage * self.goods_value) + np.sum(np.sum(np.dot(self.route_selection, self.goods_value), axis=2) * self.transit_duty)

        model.minimize(transport_cost + warehouse_cost + tax_cost)

        # --- Add Constraints ---
        self._add_common_constraints(model, arrival_time)
        self.model = model

    def solve(self, solver=None) -> None:
        """
        Solves the optimization model and stores the results.

        Args:
            solver: The solver to use (primarily for CVXPY). If None, a default
                    solver is used.
        """
        try:
            if self.framework == 'CVXPY':
                self.objective_value = self.model.solve(solver=solver or cp.CBC)
                # ... (extract solution values)
            else:  # DOCPLEX
                solution = self.model.solve()
                if not solution:
                    raise Exception("Model is not feasible or could not be solved.")
                self.objective_value = self.model.objective_value
                # ... (extract solution values)

            self._process_solution()

        except Exception as e:
            print(f"An error occurred during model solving: {e}")
            raise

    def _process_solution(self) -> None:
        """Processes the raw solution from the solver into a user-friendly format."""
        # ... (logic for processing the solution remains largely the same)

    def generate_solution_text(self, order_data: pd.DataFrame, route_data: pd.DataFrame) -> str:
        """
        Generates a human-readable text summary of the optimization solution.

        Args:
            order_data (pd.DataFrame): The original order data.
            route_data (pd.DataFrame): The original route data.

        Returns:
            str: A formatted string detailing the solution.
        """
        travel_mode_map = dict(zip(zip(route_data['Source'], route_data['Destination']), route_data['Travel Mode']))
        
        summary = [
            "====================================",
            "  Multi-Modal Transport Solution  ",
            "====================================",
            f"Number of goods: {len(order_data)}",
            f"Total cost: {self.objective_value:.2f}",
            f"  - Transportation cost: {self.final_transport_cost:.2f}",
            f"  - Warehouse cost: {self.final_warehouse_cost:.2f}",
            f"  - Tax cost: {self.final_tax_cost:.2f}",
            "------------------------------------"
        ]

        for i in range(len(order_data)):
            goods_id = f"Goods-{i+1}"
            commodity = order_data['Commodity'].iloc[i]
            start_date = pd.to_datetime(order_data['Order Date'].iloc[i]).date().isoformat()
            arrival_date = self.arrival_times[goods_id]
            
            summary.append(f"{goods_id}  |  Category: {commodity}")
            summary.append(f"Start date: {start_date}  |  Arrival date: {arrival_date}")
            summary.append("Optimal Route:")
            
            route_details = self.solution_[goods_id]
            for idx, leg in enumerate(route_details):
                from_loc, to_loc, date, _ = leg
                mode = travel_mode_map.get((from_loc, to_loc), "Unknown")
                summary.append(f"  ({idx+1}) Date: {date} | From: {from_loc} | To: {to_loc} | By: {mode}")
            summary.append("------------------------------------")
            
        return "\n".join(summary)


def preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads and preprocesses the route and order data from an Excel file.

    Args:
        file_path (str): The path to the 'model data.xlsx' file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the processed
                                           order and route DataFrames.
    """
    order_info = pd.read_excel(file_path, sheet_name='Order Information')
    route_info = pd.read_excel(file_path, sheet_name='Route Information')

    # --- Preprocess Order Data ---
    order_info.loc[order_info['Journey Type'] == 'Domestic', 'Tax Percentage'] = 0

    # --- Preprocess Route Data ---
    cost_columns = ['Port/Airport/Rail Handling Cost', 'Bunker/ Fuel Cost', 
                    'Documentation Cost', 'Equipment Cost', 'Extra Cost']
    time_columns = ['CustomClearance time (hours)', 'Port/Airport/Rail Handling time (hours)',
                    'Extra Time', 'Transit time (hours)']
    
    route_info['Cost'] = route_info[cost_columns].sum(axis=1)
    route_info['Time'] = np.ceil(route_info[time_columns].sum(axis=1) / 24)
    
    # ... (rest of the preprocessing logic remains the same)

    return order_info, route_info


if __name__ == '__main__':
    # --- 1. Load and Preprocess Data ---
    try:
        order_df, route_df = preprocess_data("model data.xlsx")
    except FileNotFoundError:
        print("Error: 'model data.xlsx' not found. Please ensure the file is in the correct directory.")
        exit()

    # --- 2. Initialize and Build the Model ---
    # To use the open-source CVXPY framework, uncomment the following line:
    # optimizer = MultiModalOptimizer(framework='CVXPY')
    optimizer = MultiModalOptimizer(framework='DOCPLEX')
    
    optimizer.set_parameters(route_df, order_df)
    optimizer.build_model()

    # --- 3. Solve the Model ---
    print("Solving the optimization problem... (This may take a few moments)")
    optimizer.solve()

    # --- 4. Generate and Save the Solution ---
    solution_text = optimizer.generate_solution_text(order_df, route_df)
    
    with open("Solution_revamped.txt", "w") as text_file:
        text_file.write(solution_text)
        
    print("\nOptimization complete. Solution saved to 'Solution_revamped.txt'")
    print(solution_text)