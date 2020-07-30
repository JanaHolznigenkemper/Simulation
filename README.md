# Simulation of a Job Shop Manufacturing System
Code and data of the experiments of the paper: *Robust Shipping Date Predictions in Manufacturing using High-level Petri Nets*
## Test scenario: 
- 4 level production manufacturing 
- Test set with real anonymised orders: 
- You can create your own test set, where e.g. buffer times are included.  

## Required data: 
- production times per machine and operation: production_data_anonymised.csv
- disruption times per machine and operation: disruption_data_anonymised.csv
- material data (we use material and operation synonymously: Material_anonymised.csv
- example of anonymised scheduled orders, with an buffer of 60 seconds after each operation: example_orders_anonymous.pickle

To run one simulation at once only, use the script: simulation.ipynb
For multiple parallel simulations use: parallelised_simulation.ipynb 
