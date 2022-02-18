import matplotlib
import static_STRATA
import sys
import matplotlib.pyplot as plt
import random

# Testing function generations
strata = static_STRATA.static_STRATA()
x = []
y = []

strata.tasks["M1"].trait_relationships["U1"].generate_random_power_relationship(max_u=1, min_m=0, max_m=1)
for i in range(1000):
    xi = random.random()
    x.append(xi)
    y.append(strata.tasks["M1"].trait_relationships["U1"].calc_relationship_performance(xi))

plt.scatter(x, y)
plt.show()


sys.exit()

### TRIAL 1: sanity check ###
strata = static_STRATA.static_STRATA()
#strata.tasks["M1"].trait_relationships["U1"].generate_random_power_relationship(max_u=1, min_m=0, max_m=1)
strata.update()
print("Q")
strata.print_Q()
print("Y*")
strata.print_Y(Y=strata.Y_s)
strata.solve()
p_opt, _ = strata.calc_performance(optimal=True)
print("P Optimal", p_opt)
p_strata, y_strata = strata.calc_performance(X=strata.X)
print("X STRATA")
strata.print_X()
print("P STRATA", p_strata)
print("error STRATA", strata.calc_performance_error(p=p_strata))
X_complete, p_complete = strata.complete_solve()
print("X Complete")
strata.print_X(X=X_complete)
print("P Complete", p_complete)
print("error Complete", strata.calc_performance_error(p=p_complete))

### TRIAL 2: random ###
