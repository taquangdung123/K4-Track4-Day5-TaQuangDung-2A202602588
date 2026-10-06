import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import hashlib

def make_F(dt):
    """State transition matrix for 2D position + velocity"""
    # Input: dt (float) - time step
    # Output: F (4×4 ndarray)
    # x[t+1] = F @ x[t] + w  (w ~ N(0, Q))
    
    F = np.eye(4)
    F[0, 2] = dt  # x += vx·dt
    F[1, 3] = dt  # y += vy·dt
    return F

def make_H():
    """Measurement matrix for GPS/UWB (measure [x, y] only)"""
    # Input: None
    # Output: H (2×4 ndarray)
    # z = H @ x  (we measure x, y only)
    
    H = np.zeros((2, 4))
    H[0, 0] = 1  # z[0] = x
    H[1, 1] = 1  # z[1] = y
    return H

def make_Q(dt, q):
    """Process noise covariance (random acceleration model)"""
    # Input: dt, q (scalar)
    # Output: Q (4×4, positive definite)
    # Model: assume random acceleration with magnitude q
    
    # Exact: Q = integral of phi(τ) @ G @ G.T @ phi(τ).T dτ
    # where phi is state transition, G is noise input
    # Simplified: diagonal with dt²/2 terms for position, dt for velocity
    
    Q = np.zeros((4, 4))
    Q[0, 0] = (dt**4 / 4) * q
    Q[0, 2] = (dt**3 / 2) * q
    Q[1, 1] = (dt**4 / 4) * q
    Q[1, 3] = (dt**3 / 2) * q
    Q[2, 0] = (dt**3 / 2) * q
    Q[2, 2] = dt * q
    Q[3, 1] = (dt**3 / 2) * q
    Q[3, 3] = dt * q
    return Q


class KalmanFilter:
    """A standard linear Kalman filter over an arbitrary-dimension state."""
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)      # state mean
        self.P = np.array(P0, dtype=float)      # state covariance

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        y = z - H @ self.x                         # innovation
        S = H @ self.P @ H.T + R                   # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return y, S, K

def run_fusion(meas, q, x0, P0):
    """Fuse asynchronous measurements from several sensors with one Kalman filter.

    Args:
        meas: list of (ts, name, z, H, R) tuples.
        q: process noise parameter.
        x0: initial state (4,).
        P0: initial covariance (4x4).

    Returns:
        list of (ts, x, P) -- one entry per measurement, in time order.
    """
    meas = sorted(meas, key=lambda m: m[0])
    kf = KalmanFilter(x0, P0)
    t_prev = meas[0][0]
    log = []
    for ts, name, z, H, R in meas:
        dt = ts - t_prev
        if dt > 0:                                   # same timestamp -> skip predict
            kf.predict(make_F(dt), make_Q(dt, q))
        kf.update(z, H, R)                           # uses THIS sensor's H and R
        t_prev = ts
        log.append((ts, kf.x.copy(), kf.P.copy()))
    return log

def gated_update(kf, z, H, R, p=0.99):
    """Apply a chi-squared gated Kalman update, rejecting outliers.

    Args:
        kf: A `KalmanFilter` instance, updated in place if the measurement
            is accepted.
        z: Measurement vector.
        H: Measurement matrix.
        R: Measurement noise covariance.
        p: Gate probability; a measurement is rejected if its squared
            Mahalanobis distance exceeds the chi-squared `p`-quantile.

    Returns:
        bool: True if the measurement was accepted and the filter updated,
        False if it was rejected as an outlier (filter left unchanged).
    """
    y = z - H @ kf.x                          # innovation
    S = H @ kf.P @ H.T + R                    # innovation covariance
    d2 = float(y @ np.linalg.solve(S, y))     # squared Mahalanobis distance
    if d2 > chi2.ppf(p, df=len(z)):           # outlier -> reject, filter unchanged
        return False
    kf.update(z, H, R)
    return True

def run_fusion_gated(meas, q, x0, P0, p=0.99):
    """Run `run_fusion`-style tracking, with outlier gating on updates.

    Args:
        meas: List of `(timestamp, name, z, H, R)` tuples, sorted by time.
        q: Process noise strength passed to `make_Q`.
        x0: Initial state estimate.
        P0: Initial state covariance.
        p: Gate probability, forwarded to `gated_update`.

    Returns:
        tuple: `(log, accepted)` where `log` is the `(timestamp, x, P)`
        list (as in `run_fusion`) and `accepted` is a boolean array marking
        which measurements passed the gate.
    """
    kf, t_prev, log, accepted = KalmanFilter(x0, P0), 0.0, [], []
    for ts, name, z, H, R in meas:
        if ts - t_prev > 0:
            kf.predict(make_F(ts - t_prev), make_Q(ts - t_prev, q))
        accepted.append(gated_update(kf, z, H, R, p))
        t_prev = max(t_prev, ts)
        log.append((ts, kf.x.copy(), kf.P.copy()))
    return log, np.array(accepted)

def generate_mission_log(student_id, T=90.0, dt_true=0.05, _reveal_truth=False):
    """Generate one personal sensor log for the Lynx-07 test vehicle.

    Every student gets a different (but reproducible) mission: a different
    path, and a different hidden sensor fault. `_reveal_truth=True` is for
    instructor grading only -- flipping it yourself only cheats you out of
    the actual exercise (diagnosing a fault you can't see is the point).

    Args:
        student_id: The student's ID or name; determines the random seed
            for this mission (see `seed_from_id`).
        T: Mission duration, in seconds.
        dt_true: Time resolution of the internally simulated ground truth.
        _reveal_truth: If True, also return the hidden ground truth and
            fault parameters. Instructor use only.

    Returns:
        dict: If `_reveal_truth` is False (the default), the `public` dict
        with sensor specs and measurements (`SPEC`, `t_gps`, `z_gps`,
        `t_uwb`, `z_uwb`, `T`).
        tuple: If `_reveal_truth` is True, `(public, truth)` where `truth`
        additionally contains the true path and the injected fault
        parameters (`fault_sensor`, `fault_type`, `bias_vec`, `noise_mult`,
        `burst_start`, `burst_len`).
    """
    seed = seed_from_id(student_id)
    rng = np.random.default_rng(seed)
    tt, true_path = _true_mission_path(rng, T, dt_true)

    SPEC = {"GPS": dict(rate=5.0, pos_sd=0.5), "UWB": dict(rate=10.0, pos_sd=1.0)}
    fault_sensor = rng.choice(["GPS", "UWB"])
    fault_type = rng.choice(["bias", "underrated_noise", "outlier_burst"])
    bias_vec = rng.normal(0, 1.0, 2) * rng.choice([-1, 1], 2) * 2.5 if fault_type == "bias" else np.zeros(2)
    noise_mult = rng.uniform(2.5, 4.5) if fault_type == "underrated_noise" else 1.0
    burst_start = rng.uniform(25, T - 20); burst_len = rng.uniform(4, 9); burst_mag = rng.uniform(8, 20)

    def make_sensor(name):
        """Simulate one sensor's noisy measurement stream for this mission.

        Args:
            name: Sensor name, `"GPS"` or `"UWB"`.

        Returns:
            tuple: `(tc, z)`, the capture times and noisy position
            measurements for this sensor, including any injected fault.
        """
        rate, sd = SPEC[name]["rate"], SPEC[name]["pos_sd"]
        tc = np.arange(0, T, 1.0 / rate)
        idx = np.clip((tc / dt_true).astype(int), 0, len(tt) - 1)
        true_xy = true_path[idx, :2]
        eff_sd = sd * (noise_mult if fault_sensor == name and fault_type == "underrated_noise" else 1.0)
        z = true_xy + rng.normal(0, eff_sd, true_xy.shape)
        if fault_sensor == name and fault_type == "bias":
            z = z + bias_vec
        if fault_sensor == name and fault_type == "outlier_burst":
            in_burst = (tc >= burst_start) & (tc < burst_start + burst_len)
            ang = rng.uniform(0, 2 * np.pi, int(in_burst.sum()))
            z[in_burst] += burst_mag * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return tc, z

    t_gps, z_gps = make_sensor("GPS")
    t_uwb, z_uwb = make_sensor("UWB")
    public = dict(SPEC=SPEC, t_gps=t_gps, z_gps=z_gps, t_uwb=t_uwb, z_uwb=z_uwb, T=T)
    if not _reveal_truth:
        return public
    truth = dict(path=true_path, t=tt, fault_sensor=fault_sensor, fault_type=fault_type,
                 bias_vec=bias_vec, noise_mult=noise_mult, burst_start=burst_start, burst_len=burst_len)
    return public, truth

print("Hạ tầng Phần 9 đã sẵn sàng: seed_from_id(), generate_mission_log()")

def diagnose(mission, q=0.3):
    """Run a provisional filter that trusts SPEC, and log diagnostics.

    Args:
        mission: A mission dict as returned by `generate_mission_log`.
        q: Process noise strength assumed for the provisional filter.

    Returns:
        dict: Keyed by sensor name (`"GPS"`, `"UWB"`), each value a dict
        with `nis` (array of per-measurement NIS values) and `resid_mean`
        (the mean 2-D residual vector), for you to inspect for signs of a
        sensor fault.
    """
    H = make_H()
    meas = sorted([(t, "GPS", z) for t, z in zip(mission["t_gps"], mission["z_gps"])] +
                  [(t, "UWB", z) for t, z in zip(mission["t_uwb"], mission["z_uwb"])], key=lambda r: r[0])
    kf = KalmanFilter([0, 0, 0, 0], np.diag([50., 50., 20., 20.]))
    t_prev = 0.0
    log = {"GPS": {"nis": [], "resid": []}, "UWB": {"nis": [], "resid": []}}
    for t, name, z in meas:
        dt = t - t_prev
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        R = np.eye(2) * mission["SPEC"][name]["pos_sd"] ** 2
        y, S, K = kf.update(z, H, R)
        nis = float(y @ np.linalg.solve(S, y))
        log[name]["nis"].append(nis)
        log[name]["resid"].append(y)
        t_prev = t
    out = {}
    for name in ["GPS", "UWB"]:
        nis = np.array(log[name]["nis"]); resid = np.array(log[name]["resid"])
        out[name] = dict(nis=nis, resid_mean=resid.mean(axis=0))
    return out

diag = diagnose(mission)
for name in ["GPS", "UWB"]:
    d = diag[name]
    print(f"{name}: n={len(d['nis'])}  mean(NIS)={d['nis'].mean():.2f}  "
          f"median(NIS)={np.median(d['nis']):.2f}  mean residual={np.round(d['resid_mean'], 2)}")


def apply_mission_fix(diag, fix_sensor, fix_method):
    """Turn a diagnosis label into filter kwargs. Students do not tune numbers by hand.

    Args:
        diag: Output of `diagnose`.
        fix_sensor: `"GPS"` or `"UWB"`.
        fix_method: `"bias"`, `"inflate_R"`, or `"gate"`.

    Returns:
        dict: Keyword arguments for `run_mission_filter`.
    """
    kwargs = dict(gps_bias_corr=np.zeros(2), uwb_bias_corr=np.zeros(2),
                  gps_r_mult=1.0, uwb_r_mult=1.0, use_gate=False, gate_sensor=None)
    if fix_method == "bias":
        key = "gps_bias_corr" if fix_sensor == "GPS" else "uwb_bias_corr"
        kwargs[key] = np.asarray(diag[fix_sensor]["resid_mean"], dtype=float)
    elif fix_method == "inflate_R":
        # 2-D NIS has expectation ~2 when R is honest. Scale that sensor's R up to match.
        key = "gps_r_mult" if fix_sensor == "GPS" else "uwb_r_mult"
        kwargs[key] = max(2.0, float(np.mean(diag[fix_sensor]["nis"])) / 2.0)
    elif fix_method == "gate":
        kwargs["use_gate"] = True
        kwargs["gate_sensor"] = fix_sensor
    else:
        raise ValueError("FIX_METHOD không hợp lệ")
    return kwargs

def run_mission_filter(mission, q=0.3,
                       gps_bias_corr=np.zeros(2), uwb_bias_corr=np.zeros(2),
                       gps_r_mult=1.0, uwb_r_mult=1.0,
                       use_gate=False, gate_sensor=None, gate_p=0.999):
    """Fuse GPS and UWB for the whole mission, with one optional fault correction.

    Args:
        mission: A mission dict as returned by `generate_mission_log`.
        q: Process noise strength passed to `make_Q`.
        gps_bias_corr: Bias subtracted from GPS readings before the update.
        uwb_bias_corr: Bias subtracted from UWB readings before the update.
        gps_r_mult: Multiplier on GPS measurement variance.
        uwb_r_mult: Multiplier on UWB measurement variance.
        use_gate: If True, chi-squared gate measurements (see `gate_sensor`).
        gate_sensor: If set, only this sensor is gated. Other sensors always update.
        gate_p: Gate probability forwarded to `gated_update`.

    Returns:
        tuple: `(log, nis_all)` of accepted updates. `gated_update` returns a bool;
        NIS is computed here and is not unpacked from `gated_update`.
    """
    H = make_H()
    meas = sorted([(t, "GPS", z) for t, z in zip(mission["t_gps"], mission["z_gps"])] +
                  [(t, "UWB", z) for t, z in zip(mission["t_uwb"], mission["z_uwb"])], key=lambda r: r[0])
    kf = KalmanFilter([0, 0, 0, 0], np.diag([50., 50., 20., 20.]))
    t_prev, log, nis_all = 0.0, [], []
    bias = {"GPS": np.asarray(gps_bias_corr, dtype=float), "UWB": np.asarray(uwb_bias_corr, dtype=float)}
    r_mult = {"GPS": gps_r_mult, "UWB": uwb_r_mult}
    for t, name, z in meas:
        dt = t - t_prev
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        zc = np.asarray(z, dtype=float) - bias[name]
        R = np.eye(2) * mission["SPEC"][name]["pos_sd"] ** 2 * r_mult[name]
        gate_this = use_gate and (gate_sensor is None or name == gate_sensor)
        if gate_this:
            y = zc - H @ kf.x
            S = H @ kf.P @ H.T + R
            nis = float(y @ np.linalg.solve(S, y))
            ok = gated_update(kf, zc, H, R, p=gate_p)
            if not ok:
                t_prev = t
                continue
        else:
            y, S, K = kf.update(zc, H, R)
            nis = float(y @ np.linalg.solve(S, y))
        log.append((t, kf.x.copy(), kf.P.copy()))
        nis_all.append(nis)
        t_prev = t
    return log, np.array(nis_all)

kw = apply_mission_fix(diag, FIX_SENSOR, FIX_METHOD)
log, nis_all = run_mission_filter(mission, **kw)
print(f"Pooled mean NIS = {nis_all.mean():.2f}  (median = {np.median(nis_all):.2f})  "
      f"trên {len(nis_all)} phép đo được chấp nhận")



STUDENT_ID = '2A202602588'
mission = generate_mission_log(STUDENT_ID)
diag = diagnose(mission)
print('DIAGNOSE:')
for k, v in diag.items():
    print(k, 'mean NIS:', np.mean(v['nis']), 'median NIS:', np.median(v['nis']), 'resid_mean:', v['resid_mean'])

# Let us check both fixes: UWB bias vs GPS bias
for fix_s, fix_m in [('UWB', 'bias'), ('GPS', 'bias'), ('UWB', 'inflate_R'), ('UWB', 'gate')]:
    kw = apply_mission_fix(diag, fix_s, fix_m)
    log, nis_all = run_mission_filter(mission, **kw)
    last_P = log[-1][2]
    sig_1 = np.sqrt(np.trace(last_P[:2, :2]))
    print(f'Fix {fix_s} {fix_m}: pooled mean NIS = {nis_all.mean():.4f}, median = {np.median(nis_all):.4f}, 1sigma = {sig_1:.4f}, last_P trace = {np.trace(last_P[:2, :2]):.4f}')
