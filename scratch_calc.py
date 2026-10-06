import hashlib

def seed_from_id(student_id: str) -> int:
    """Derive a reproducible personal random seed from a student ID.

    Args:
        student_id: The student's ID or name, as a string.

    Returns:
        int: A deterministic seed derived from the SHA-256 hash of the
        (lowercased, stripped) `student_id`. The same ID always yields the
        same seed; different IDs yield different (effectively random) seeds.
    """
    h = hashlib.sha256(student_id.strip().lower().encode()).hexdigest()
    return int(h[:8], 16)

def _true_mission_path(rng, T, dt_true):
    """Generate a smooth, randomized ground-truth path for the mission.

    Args:
        rng: A `numpy.random.Generator`, already seeded.
        T: Mission duration, in seconds.
        dt_true: Time resolution of the generated path, in seconds.

    Returns:
        tuple: `(tt, true_path)` where `tt` is the time vector and
        `true_path` is an array of shape `(len(tt), 4)` with columns
        `[x, y, vx, vy]`.
    """
    n_wp = rng.integers(4, 7)
    wps = np.vstack([[0, 0], rng.uniform(-40, 40, size=(n_wp, 2))])
    seg_t = np.linspace(0, T, len(wps))
    tt = np.arange(0, T, dt_true)
    px = np.interp(tt, seg_t, wps[:, 0]); py = np.interp(tt, seg_t, wps[:, 1])
    k = np.ones(15) / 15
    px = np.convolve(px, k, mode="same"); py = np.convolve(py, k, mode="same")
    vx = np.gradient(px, dt_true); vy = np.gradient(py, dt_true)
    return tt, np.stack([px, py, vx, vy], axis=1)

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


