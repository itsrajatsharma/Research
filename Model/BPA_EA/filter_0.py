import matplotlib.pyplot as plt
import numpy as np

# System parameters
TR = 0.1  # Filter: 1/(1+sTR)
TA = 0.2  # Regulator: KA/(1+sTA)
TA1 = 0.05  # Additional time constant: 1/(1+sTA1)
TF = 0.05  # Stabilizer: sKF/(1+sTF)
TE = 0.3  # Exciter: KE/(sTE)
KF = 1.0  # Stabilizer gain
KA = 10.0  # Regulator gain
KE = 1.0  # Exciter gain
VRMAX = 1.2  # Jerry Input
VRMIN = 0.8  # Jerry Input

# Simulation parameters
dt = 0.005
t_max = 5.0
steps = int(t_max / dt)


class SinusoidalNoiseGenerator:
    """Generates multi-component sinusoidal noise signal"""

    def __init__(self, steps, dt, amplitude=0.1, num_components=5,
                 min_freq=1.0, max_freq=10.0, seed=None):
        self.steps = steps
        self.dt = dt
        self.amplitude = amplitude
        self.num_components = num_components
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.seed = seed
        self.noise_signal = None

    def generate(self):
        """Precompute the noise signal"""
        np.random.seed(self.seed)
        time = np.arange(self.steps) * self.dt
        freqs = np.random.uniform(self.min_freq, self.max_freq, self.num_components)
        phases = np.random.uniform(0, 2 * np.pi, self.num_components)

        self.noise_signal = np.zeros(self.steps)
        for freq, phase in zip(freqs, phases):
            self.noise_signal += np.sin(2 * np.pi * freq * time + phase)

        # Normalize to [-amplitude, amplitude]
        self.noise_signal = self.amplitude * (self.noise_signal / np.max(np.abs(self.noise_signal)))
        return self


class IIRControlComponent:
    """1st-order IIR component using standard difference equation"""

    def __init__(self, a1, a2, b1, b2):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.x_prev = 0.0
        self.y_prev = 0.0

    def update(self, x_current):
        alpha = (self.b2 - self.b1 * dt) / self.b2
        beta = self.a2 / self.b2
        gamma = (-self.a2 + self.a1 * dt) / self.b2
        y_current = alpha * self.y_prev + beta * x_current + gamma * self.x_prev
        self.x_prev = x_current
        self.y_prev = y_current
        return y_current


class FIRControlComponent:
    """1st-order FIR filter: y[n] = b0*x[n] + b1*x[n-1]"""

    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b1
        self.x_prev = 0.0

    def update(self, x_current):
        y_current = self.b0 * x_current + self.b1 * self.x_prev
        self.x_prev = x_current
        return y_current


# Initialize components with noise generator
noise_gen = SinusoidalNoiseGenerator(
    steps=steps,
    dt=dt,
    amplitude=0.15,
    num_components=5,
    min_freq=1.0,
    max_freq=10.0,
    seed=42
).generate()

filter_block = IIRControlComponent(a1=1, a2=0, b1=1, b2=TR)
regulator_ka = IIRControlComponent(a1=KA, a2=0, b1=1, b2=TA)
regulator_ta1 = IIRControlComponent(a1=1, a2=0, b1=1, b2=TA1)
stabilizer = IIRControlComponent(a1=0, a2=KF, b1=1, b2=TF)
fir_filter = FIRControlComponent(b0=0.7, b1=0.3)
exciter = IIRControlComponent(a1=KE, a2=0, b1=0, b2=TE)

# Storage for visualization
time = []
results = {
    'filter_out': [],
    'reg_ka_out': [],
    'vr_sat_out': [],
    'stabilizer_out': [],
    'exciter_out': [],
    'fir_out': [],
    'noise': []
}

# Initialize feedback variables
y_stabilizer_prev = 0.0
y_fir_prev = 0.0

for k in range(steps):
    # 1. Filter block with noisy input (ONLY CHANGED PART)
    vt = 1.0 + noise_gen.noise_signal[k]
    y_filter = filter_block.update(vt)

    # 2. Summation (V_REF - V2 + V_STB)
    v_ref = 1  # Jerry Input
    v_stb = 0  # Jerry Input
    reg_input = v_ref + v_stb - y_filter - y_stabilizer_prev

    # 3. Regulator KA
    y_reg_ka = regulator_ka.update(reg_input)

    # 4. Additional time constant
    y_reg_ta1 = regulator_ta1.update(y_reg_ka)

    # 5. Saturation
    vr_sat = max(VRMIN, min(y_reg_ta1, VRMAX))

    # 6. Summation (V_REF - V2 + V_STB)
    feedback = vr_sat - y_fir_prev

    # 7. Exciter
    y_exciter = exciter.update(feedback)
    print(y_exciter)

    # 8. Stabilizer
    y_stabilizer = stabilizer.update(y_exciter)
    y_stabilizer_prev = y_stabilizer

    # 9. FIR Filter
    y_fir = fir_filter.update(y_exciter)
    y_fir_prev = y_fir

    # Store results
    time.append(k * dt)
    results['filter_out'].append(y_filter)
    results['reg_ka_out'].append(y_reg_ka)
    results['vr_sat_out'].append(vr_sat)
    results['stabilizer_out'].append(y_stabilizer)
    results['exciter_out'].append(y_exciter)
    results['fir_out'].append(y_fir)
    results['noise'].append(noise_gen.noise_signal[k])

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(time, results['noise'], ':', alpha=0.7, label='Input Noise (15%)')
plt.plot(time, results['exciter_out'], label='Exciter: KE/(sTE) Output (E_FD)')             # Exciter: KE/(sTE)
plt.plot(time, results['filter_out'], '--', label='Filter: 1/(1+sTR) Output')               # Filter: 1/(1+sTR)
plt.plot(time, results['reg_ka_out'], '-.', label='Regulator: KA/(1+sTA) Output')           # Regulator: KA/(1+sTA)
plt.plot(time, results['vr_sat_out'], '-', label='vr_sat Output (LIMITER)')                 # Limiter
plt.plot(time, results['stabilizer_out'], ':', label='Stabilizer: sKF/(1+sTF) Output')      # Stabilizer: sKF/(1+sTF)
plt.plot(time, results['fir_out'], '-', label='FIR : SE + KE Output (b0=0.7, b1=0.3)')      # FIR : SE + KE
plt.title('BPA EA - Control System Response with Noisy Step Input')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()


