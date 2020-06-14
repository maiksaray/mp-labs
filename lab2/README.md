This folder has implementations of parallel system of linear equation solving using Jacobi method.
Parallelism achieved using MPI, MS MPI was used during benchmarkings 

There are following implementations compared (first two in _sequential): 
* sequential reference run using numpy for iteration calculations
* sequential reference run using python native math for each iteration
* parallel implementation with both numpy calculations and native calculations

