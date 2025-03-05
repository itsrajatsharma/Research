import matplotlib.pyplot as plt

# System parameters
TR = 0.1  # Filter: 1/(1+sTR)
TA = 0.2  # Regulator: KA/(1+sTA)
TA1 = 0.05  # Additional time constant: 1/(1+sTA1)
TF = 0.05  # Stabilizer: sKF/(1+sTF)
TE = 0.3  # Exciter: KE/(sTE)
KF = 1.0  # Stabilizer gain
KA = 10.0  # Regulator gain
KE = 1.0  # Exciter gain
VRMAX = 5.0
VRMIN = -5.0

# Simulation parameters
dt = 0.005
t_max = 5.0
steps = int(t_max / dt)


class ControlComponent:
    """Generic 1st-order component using the standard equation"""

    def __init__(self, a1, a2, b1, b2):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.x_prev = 0.0
        self.y_prev = 0.0

    def update(self, x_current):
        """Implements y[n] = αy[n-1] + βx[n] + γx[n-1]"""
        alpha = (self.b2 - self.b1 * dt) / self.b2
        beta = self.a2 / self.b2
        gamma = (-self.a2 + self.a1 * dt) / self.b2

        y_current = alpha * self.y_prev + beta * x_current + gamma * self.x_prev

        self.x_prev = x_current
        self.y_prev = y_current
        return y_current


class Integrator:
    """Special case for KE/(sTE) using backward Euler"""

    def __init__(self, gain, time_const):
        self.gain = gain
        self.time_const = time_const
        self.y_prev = 0.0

    def update(self, x_current):
        y_current = self.y_prev + (self.gain * dt / self.time_const) * x_current
        self.y_prev = y_current
        return y_current


# Initialize components using correct coefficients from transfer functions
filter_block = ControlComponent(a1=1, a2=0, b1=TR, b2=1)
regulator_ka = ControlComponent(a1=KA, a2=0, b1=TA, b2=1)
regulator_ta1 = ControlComponent(a1=1, a2=0, b1=TA1, b2=1)
stabilizer = ControlComponent(a1=0, a2=KF, b1=TF, b2=1)
exciter = Integrator(KE, TE)

# Storage for visualization
time = []
results = {
    'filter_out': [],
    'reg_ka_out': [],
    'reg_ta1_out': [],
    'stabilizer_out': [],
    'exciter_out': []
}

# Previous stabilizer output for feedback
y_stabilizer_prev = 0.0

for k in range(steps):
    # 1. Filter block
    vt = 1.0  # Step input
    y_filter = filter_block.update(vt)

    # 2. Summation (V_REF - V2 + V_STB)
    v_ref = 1.0
    v_stb = 1.0
    reg_input = v_ref + v_stb - y_filter + y_stabilizer_prev

    # 3. Regulator KA
    y_reg_ka = regulator_ka.update(reg_input)

    # 4. Additional time constant
    y_reg_ta1 = regulator_ta1.update(y_reg_ka)

    # 5. Saturation
    vr_sat = max(VRMIN, min(y_reg_ta1, VRMAX))

    # 6. Exciter
    y_exciter = exciter.update(vr_sat)

    # 7. Stabilizer
    y_stabilizer = stabilizer.update(y_exciter)
    y_stabilizer_prev = y_stabilizer  # Store for next iteration feedback

    # Store results
    time.append(k * dt)
    results['filter_out'].append(y_filter)
    results['reg_ka_out'].append(y_reg_ka)
    results['reg_ta1_out'].append(y_reg_ta1)
    results['stabilizer_out'].append(y_stabilizer)
    results['exciter_out'].append(y_exciter)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(time, results['exciter_out'], label='Exciter Output (E_FD)')
plt.plot(time, results['filter_out'], '--', label='Filter Output')
plt.plot(time, results['reg_ka_out'], '-.', label='Regulator KA')
plt.plot(time, results['stabilizer_out'], ':', label='Stabilizer')
plt.title('Control System Response Using General Filter Equation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
