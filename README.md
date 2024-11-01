# Numerical Analysis Part I: Assignment

## Installation Instructions
Navigate into the top level of the repository and run:
```
$ python -m pip install .
```
This should install all the necessary dependencies.

If you intend to make any changes to the Python code after installation, it may be beneficial to install the package in editable mode:
```
$ python -m pip install -e .
```


## Scripts
Scripts will be installed alongside the Python package. To run them use the following syntax:
```
$ script-name param1 param2 ...
```

To view all the parameters and options of a script, run:
```
$ script-name -h
```

For example `$ plot_convergence -h` returns:
```
usage: plot_convergence [-h] [--final_time FINAL_TIME] [--min_grid MIN_GRID] [--max_grid MAX_GRID] [--method {forward,forward_backward}] [--plot_solutions]
                        [--interpolation_points INTERPOLATION_POINTS] [--start_point START_POINT] [--end_point END_POINT] [--g G] [--theta THETA] [--H H] [--U U]
                        [--dt DT] [--path PATH] [--tag TAG]
                        initial_h initial_u

Plot the convergence of the linear shallow water equations.

positional arguments:
  initial_h             An expression in the coordinate x for the inital condition of the variable h. The function should be a quoted string. E.g. "sin(x)". The
                        function must be periodic and always return a float.
  initial_u             An expression in the coordinate x for the inital condition of the variable u. The function should be a quoted string. E.g. "sin(x)". The
                        function must be periodic and always return a float.

options:
  -h, --help            show this help message and exit
  --final_time FINAL_TIME
                        The target final time of the simulation. (default: 1.0)
  --min_grid MIN_GRID   The minimum grid size to use for the convergence study. (default: 4)
  --max_grid MAX_GRID   The maximum grid size to use for the convergence study. (default: 10)
  --method {forward,forward_backward}
                        The method to use for the simulation. Selecting "forward_backward" will override the U parameter. (default: forward_backward)
  --plot_solutions      Plot the solutions. (default: False)
  --interpolation_points INTERPOLATION_POINTS
                        The number of points to use for the line of best fit. (default: 3)
  --start_point START_POINT
                        The start point of the domain. (default: 0.0)
  --end_point END_POINT
                        The end point of the domain. (default: 1.0)
  --g G                 The acceleration due to gravity. (default: 9.81)
  --theta THETA         The angle of the bed. (default: 0.0)
  --H H                 The reference depth of the fluid. (default: 1.0)
  --U U                 The reference velocity of the fluid. (default: 1.0)
  --dt DT               An expression in the parameters dx, U, H, g for the time-step dt. The function should be a quoted string. E.g. "dx**2". The function must
                        be periodic and always return a float. (default: dx)
  --path PATH           The path to save the figure to. (default: ./)
  --tag TAG             A tag to add to the saved figure. (default: )
```
The syntax for positional arguments which represent functions, such as `initial_h` and `initial_u`, you are permitted access to all mathematical functions and symbols defined by the C standard (or equivalently the standard Python `math` package).


## Reproduce Figures
To reproduce all the figures included in the report run one of the following scripts depending on your platform:

### Unix (Linux/MacOS)
To plot and save all of the figures to the current working directory, run:
```
$ ./plot_all.sh
```

Note that this may require additional permissions which can be added by running:
```
$ chmod +x plot_all.sh
```

Alternatively, you can run the script with only read permission by passing it as an argument to the `sh` command:
```
$ sh plot_all.sh
```

### Windows
To plot and save all of the figures to the current working directory, run:
```
> plot_all.cmd
```
It may be neccessary to run this command in a Command Prompt that is in administrator mode.
