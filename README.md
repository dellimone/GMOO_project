# Hybrid Optimization Algorithms: A Comparative Study

This project implements and compares several hybrid optimization algorithms, combining the strengths of Genetic Algorithms (GA) and Particle Swarm Optimization (PSO). 
The following algorithms are included:

* Sequential GA-PSO (SGAPSO): Runs GA followed by PSO.
* Alternating GA-PSO (AGAPSO): Alternates between GA and PSO phases.
* Parallel GA-PSO (PGAPSO): Applies GA and PSO to separate subpopulations in parallel.
* Island GA-PSO (IGAPSO): Uses multiple PSO islands with periodic migration of solutions using GA operators.

## Project Structure

```bash
├── img
│   ├── SGAPSO.svg
│   ├── AGAPSO.svg
|   ├── PGAPSO.svg
│   └── IGAPSO.svg
├── src
│   ├── benchmarks.py
|   ├── plots.py
│   ├── util.py
│   ├── ga.py
│   ├── pso.py
│   ├── sgapso.py
│   ├── agapso.py
│   ├── pagapso.py
│   └── igapso.py
├── GMOO_project.ipynb
└── README.md
```

## Dependencies

* Python 3
* NumPy
* Matplotlib

## Showcase

The [`GMOO_Project.ipynb`](GMOO_Project.ipynb) notebook provides a detailed description of the implemented algorithms and a comparison of their performance on various benchmark functions.  This notebook serves as the primary showcase for the project's results and analysis.

## Project Structure

* [`agapso.py`](src/agapso.py): Implementation of the Alternating GA-PSO algorithm.
* [`benchmarks.py`](src/benchmarks.py): Definitions of various benchmark functions.
* [`ga.py`](src/ga.py): Implementation of the Genetic Algorithm.
* [`igapso.py`](src/igapso.py): Implementation of the Island GA-PSO algorithm.
* [`pgapso.py`](src/pgapso.py): Implementation of the Parallel GA-PSO algorithm.
* [`plots.py`](src/plots.py): Script for running experiments and generating plots.
* [`pso.py`](src/pso.py): Implementation of the Particle Swarm Optimization algorithm.
* [`sgapso.py`](src/sgapso.py): Implementation of the Sequential GA-PSO algorithm.
* [`util.py`](src/util.py): Utility classes and data structures.

## Resources from literature

- [Hybrid PSO and GA for Global Maximization,K. Premalatha and A.M. Natarajan](https://www.emis.de/journals/IJOPCM/Vol/09/IJOPCM(vol.2.4.12.D.9).pdf)
- [A hybrid PSO-GA algorithm for constrained optimization problems, Harish Garg](https://doi.org/10.1016/j.amc.2015.11.001)
- [A hybrid genetic algorithm and particle swarm
optimization for multimodal functions,Yi-Tung Kao Erwie Zahara](http://dx.doi.org/10.1016/j.asoc.2007.07.002)
- [Particle Swarm Optimization: A Survey of Historical
and Recent Developments with
Hybridization Perspectives,Saptarshi Sengupta, Sanchita Basak and Richard Alan Peters](https://doi.org/10.3390/make1010010)

_This project is part of the Global and Multi-objective Optimization._
