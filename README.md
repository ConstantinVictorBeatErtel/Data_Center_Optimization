# Data Center Network Optimizer

A decision support tool built with Python and Streamlit to optimize the geographic placement of data centers. This application uses a two-stage Linear Programming (LP) model to balance physical constraints (latency, power capacity) with conflicting objectives (minimizing cost vs. minimizing carbon footprint) while ensuring population coverage.

## Project Overview

As demand for low-latency digital services grows, selecting optimal data center locations becomes a complex multi-objective problem. This dashboard allows users to:
* Visualize candidate sites across the US.
* Dynamically adjust constraints (Latency limits, GW capacity).
* Analyze the trade-off between **Economic Cost** (Energy prices) and **Environmental Impact** (CO2 emissions).

## Key Features

* **Two-Stage Optimization Engine:**
    * **Stage 1 (Feasibility):** Maximizes the total theoretical population reachable given physical constraints.
    * **Stage 2 (Selection):** Selects the optimal subset of sites to minimize a weighted function of Cost and CO2, while meeting a specific population target.
* **Interactive Scenario Planning:** Adjust the "CO2 Penalty Weight" to see how the network shifts from cheap energy states to low-carbon states.
* **Geospatial Visualization:** Interactive Plotly map showing data center locations and their served population clusters.
* **Real-time Metrics:** Dynamic calculation of Average Latency, Avg Cost ($/kWh), Avg CO2 (lb/MWh), and Total Served Population.

## Technology Stack

* **Frontend:** Streamlit
* **Optimization:** PuLP (Linear Programming)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express / Graph Objects

## Optimization Methodology

The model utilizes the **PuLP** library to solve the facility location problem.

### Inputs
* **Candidate Sites:** Filtered by minimum Energy Capacity (GW).
* **Demand Points:** US Cities with associated population data.
* **Latency Matrix:** Calculated using the Haversine formula ($R = 3958.8$ miles) converted to milliseconds (approx. 1ms per 50 miles).

### The Algorithm
1.  **Stage 1 (Maximize Reach):**
    * *Objective:* Maximize total population served.
    * *Constraint:* Max $k$ centers allowed.
    * *Result:* Calculates the `Max_Possible_Population`.
2.  **Stage 2 (Minimize Cost + Impact):**
    * *Objective:* Minimize $\sum (Cost_{raw} + \alpha \times CO2_{emissions})$.
    * *Constraint:* Must serve $\ge X\%$ of `Max_Possible_Population` (or a manual target).
    * *Result:* Returns optimal site selection and city assignment.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/datacenter-optimizer.git](https://github.com/yourusername/datacenter-optimizer.git)
    cd datacenter-optimizer
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit pandas numpy pulp plotly
    ```

3.  **Prepare Data:**
    Ensure `240_Project_Master_File.csv` is in the root directory. The CSV requires the following columns:
    * `City` (or `City / Metropolitan Area`)
    * `GW` (Realistic GW per City)
    * `Population` (Metro Pop)
    * `Cost_SD` (Standardized Energy Cost)
    * `CO2_SD` (Standardized CO2 Emissions)
    * `Cost_Raw` (Actual Cents/KWH)
    * `CO2_Raw` (Actual lb/MWh)
    * `Latitude`, `Longitude`
    * `State`

4.  **Run the Dashboard:**
    ```bash
    streamlit run 240_dashboard.py
    ```

## Usage Guide

1.  **Physical Constraints (Sidebar):**
    * **Max Number of Centers:** Cap the capital expenditure by limiting sites.
    * **Min Energy Capacity:** Filter out cities that cannot support hyperscale power requirements.
    * **Max Latency:** Set the strict radius for service (ms).
2.  **Optimization Goals:**
    * **CO2 Penalty Weight:** Slide this to `0` to optimize purely for profit. Slide to `10` to prioritize green energy regardless of cost.
3.  **Target Strategy:**
    * **Percentage (Slack):** Recommended. Finds the theoretical max coverage, then optimizes to hit, for example, 95% of that max.
    * **Manual:** Force the model to find a specific number of people (e.g., "Serve 100 Million").

## License

This project is open-source and available under the MIT License.
