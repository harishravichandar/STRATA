import matplotlib
import static_STRATA
import sys
import matplotlib.pyplot as plt
import random

# Testing function generations
strata = static_STRATA.static_STRATA()

strata.generate_traits_random(max_u=5)
strata.generate_relationships_random()
strata.update()

strata.plot_relationships()

### TRIAL 1: sanity check ###
print("Q")
strata.print_Q()
print("Y*")
strata.print_Y(Y=strata.Y_s)
strata.solve_strata()
p_opt, _ = strata.calc_performance(optimal=True)
print("P Optimal", p_opt)
p_strata, y_strata = strata.calc_performance(X=strata.X)
print("X STRATA")
strata.print_X()
print("P STRATA", p_strata)
print("error STRATA", strata.calc_performance_error(p=p_strata))
X_complete, p_complete = strata.solve_performance()
print("X Complete")
strata.print_X(X=X_complete)
print("P Complete", p_complete)
print("error Complete", strata.calc_performance_error(p=p_complete))

### TRIAL 2: random ###
