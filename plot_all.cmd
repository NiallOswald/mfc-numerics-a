@echo off

REM Produce and save figures included in the report
REM Figure 1
echo. Figure 1: Plotting solutions for FTCS
plot_solution "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx" --theta 1.0 --time_values 0.0 0.125 0.25 --grid_size 100 --method "forward" --tag "_fig1"

REM Figure 2
echo. Figure 2: Plotting convergence for FTCS
plot_convergence "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx**2" --theta 1.0 --final_time 1.0 --max_grid 9 --method "forward" --tag "_fig2"

REM Figure 3
echo. Figure 3: Plotting solutions at long times for FTCS
plot_solution "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx**2" --theta 1.0 --time_values 1.0 3.75 4.0 --grid_size 100 --method "forward" --tag "_fig3"

REM Figure 4
echo. Figure 4: Plotting solutions at long times for forward-backward
plot_solution "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx / sqrt(g * H)" --theta 1.0 --time_values 1.0 2.5 5.0 --grid_size 100 --tag "_fig4"

REM Figure 5
echo. Figure 5: Plotting convergence for forward-backward
plot_convergence "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx / sqrt(g * H)" --theta 1.0 --final_time 1.0 --max_grid 14 --tag "_fig5"

REM Figure 6
echo. Figure 6: Plotting stability of forward-backward
plot_stability_sweep "cos(2 * pi * x)" "sin(2 * pi * x)" --theta 1.0 --final_time 1.0 --beta_start 1.5 --beta_end 2.1 --beta_steps 100 --tag "_fig6"

REM Figure 7
echo. Figure 7: Plotting convergence comparison
plot_comparison "cos(2 * pi * x)" "sin(2 * pi * x)" --dt "dx**2" --theta 1.0 --final_time 1.0 --max_grid 9 --tag "_fig7"
