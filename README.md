Advanced Multi-Modal Logistics Optimizer
========================================

This project provides a robust, production-ready framework for solving complex multi-modal transportation problems using mathematical programming. It is designed as a high-performance decision support tool to achieve significant cost reduction in supply chain and logistics networks by determining the optimal routing and consolidation strategy for multiple goods.

The model is architected to be flexible, supporting both the industry-leading IBM CPLEX solver (via the DOCPLEX API) and various open-source solvers through the CVXPY framework.

<p align="center"><img src="https://user-images.githubusercontent.com/30411828/45585955-c6311e80-b920-11e8-95c9-bc90089446b4.jpg"></p>

* * *

Core Features
-------------

*   **End-to-End Cost Minimization:** The objective function holistically minimizes total logistics cost, including variable transportation, fixed freight charges, warehousing/inventory holding costs, and international tariffs/duties.
    
*   **Dynamic Route & Mode Selection:** Intelligently selects the optimal path and transportation mode (Sea, Air, Rail, Truck) for each shipment based on cost, transit time, and scheduling constraints.
    
*   **Goods Consolidation:** Automatically identifies opportunities to consolidate shipments along common routes to optimize container utilization and reduce fixed costs.
    
*   **Constraint-Based Optimization:** Enforces critical business rules, including delivery deadlines, origin/destination requirements, and network flow conservation.
    
*   **Flexible Solver Integration:** Seamlessly switch between the high-performance commercial solver IBM CPLEX and accessible open-source solvers like CBC.
    

Problem Domain
--------------

In today's global supply chain, logistics managers face the challenge of shipping multiple orders from various origins to diverse destinations under tight deadlines and budgets. The network of available routes—spanning air, sea, rail, and ground—presents a combinatorial explosion of possibilities, each with unique cost structures, transit times, and schedules.

This model addresses this challenge by formulating it as a **Mixed-Integer Linear Program (MILP)**. It moves beyond simple shortest-path algorithms by considering the interplay between all shipments simultaneously, ensuring a globally optimal solution for the entire system, not just individual orders.

Technical Architecture
----------------------

The optimization model is built upon a four-dimensional decision variable, X\_ijtk, which represents the decision to ship good _k_ from node _i_ to node _j_ at time _t_. This core variable, along with supporting variables for container counts and route usage, allows the model to precisely map out the journey of every item through the logistics network.

*   **Objective Function:** The model's objective is to minimize a linear combination of:
    
    1.  **Transportation Costs:** Variable costs per container and fixed costs for utilizing a route.
        
    2.  **Warehouse Costs:** Inventory holding costs calculated based on the duration goods are stored at intermediate nodes.
        
    3.  **Tariffs & Duties:** Import taxes and transit duties applied based on the value of goods and the countries they traverse.
        
*   **Key Constraints:** The solution space is defined by a set of robust constraints that mirror real-world logistics operations:
    
    1.  **Flow Conservation:** Ensures that goods entering an intermediate node must also depart, maintaining path integrity from origin to destination.
        
    2.  **Deadline Adherence:** Guarantees that the final arrival time for each good is on or before its required delivery date.
        
    3.  **Containerization Logic:** Calculates the required number of containers for consolidated shipments based on total volume, linking routing decisions to transport capacity.
        
    4.  **Logical & Exclusivity Rules:** Prevents invalid routes, such as shipping a good back to its origin or out of its final destination.
        

Getting Started
---------------

### Prerequisites

*   Python 3.7+
    
*   Pandas, NumPy
    
*   Optimization Frameworks:
    
    *   **For CPLEX (Recommended):** `docplex` library and an installation of IBM ILOG CPLEX Optimization Studio.
        
    *   **For Open-Source:** `cvxpy`, `scipy`, and a compatible solver like `CBC` (`pip install cvxpy scs cbcpy`).
        

### Execution

1.  **Configure Data:** Populate the `Order Information` and `Route Information` sheets in the `model data.xlsx` file with your specific logistics network data.
    
2.  **Set Framework:** In `multi-modal transportation.py`, select your desired framework. The default is DOCPLEX.
    
    Python
    
        # For CPLEX
        m = MMT()
        
        # For Open-Source (e.g., CBC)
        # m = MMT(framework='CVXPY') 
3.  **Run the Model:** Execute the script from your terminal.
    
    Bash
    
        python "multi-modal transportation.py" 
4.  **Review Results:** The detailed optimal route plan and cost breakdown will be generated in `Solution.txt`.
    

Extensibility
-------------

The object-oriented design of the `MMT` class facilitates straightforward extension. The model can be adapted to incorporate more complex business logic, such as:

*   Time-window constraints for pickups and deliveries.
    
*   Stochastic variables for uncertain transit times or costs.
    
*   Multi-objective optimization (e.g., balancing cost and delivery speed).
    
*   Capacity constraints at warehouses or ports.
