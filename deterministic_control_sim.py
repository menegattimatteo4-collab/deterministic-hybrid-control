import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------
# Parameters (abstract, bounded)
# -----------------------------
P_max = 0.5
T_max = 2.2
alpha = 0.5
B_min = 0.5
r_min = 0.95

a = 0.2
b = 0.1
gamma = 0.1
beta = 0.05
Q_c_max = 0.2

# x = [P, T, dT, B, Qc, r]

def dynamics(t, x, state):
    P, T, dT, B, Qc, r = x

    if state == 2:      # Nominal
        return [0,
                a*P - b*Qc,
                0,
                -gamma*P,
                0,
                0]

    if state == 3:      # Degraded
        return [-beta*P,
                a*P - b*Qc,
                0,
                -gamma*P,
                0,
                -0.01]

    if state == 4:      # Safe
        return [0,
                -b*Q_c_max,
                0,
                0,
                0,
                0]

    return np.zeros(6)

# -----------------------------
# Event functions
# -----------------------------
def events(x):
    P, T, dT, B, Qc, r = x
    return {
        "T": T > T_max,
        "dT": dT > alpha,
        "P": P > P_max,
        "B": B < B_min,
        "r": r < r_min
    }

# -----------------------------
# Deterministic transition logic
# -----------------------------
def transition(state, ev):
    if ev["r"] or ev["T"]:
        return 4
    if state == 2 and any(ev.values()):
        return 3
    if state == 3 and (ev["r"] or ev["T"]):
        return 4
    return state

# -----------------------------
# Simulation
# -----------------------------
def simulate(Tf=100):
    x = np.array([0.4, 1.8, 0.0, 1.0, 0.0, 1.0])
    state = 2
    t = 0.0

    while t < Tf:
        sol = solve_ivp(lambda t,y: dynamics(t,y,state),
                        [t, Tf], x, max_step=0.1)
        x = sol.y[:,-1]
        ev = events(x)
        new_state = transition(state, ev)
        if new_state == state:
            break
        state = new_state
        t = sol.t[-1]

    return state, x

if __name__ == "__main__":
    s, xf = simulate(1000)
    print("Final state:", s)
    print("Final x:", xf)
