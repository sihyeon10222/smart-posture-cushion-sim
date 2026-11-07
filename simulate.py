"""Simulate the four load sensors of a seat cushion at 20 Hz."""

from dataclasses import dataclass

import numpy as np


SAMPLE_RATE_HZ = 20
SEATED_LOAD_N = 50.0
NOISE_STD_N = 0.5
RANDOM_SEED = 42
TARE_FRAMES = 5 * SAMPLE_RATE_HZ
EMA_ALPHA = 0.2
DISTANCE_LIMIT = 0.30
SPEED_LIMIT_PER_S = 0.10
HOLD_FRAMES = 15 * SAMPLE_RATE_HZ
RAMP_FRAMES = 3 * SAMPLE_RATE_HZ
PWM_MIN = 50
PWM_MAX = 200

# Coordinates in sensor order: rear-left, rear-right, front-left, front-right.
SENSOR_XY = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)


@dataclass(frozen=True)
class Scenario:
    time_s: np.ndarray
    phase: np.ndarray
    true_xy: np.ndarray
    loads_n: np.ndarray


@dataclass(frozen=True)
class Posture:
    baseline_xy: np.ndarray
    relative_xy: np.ndarray
    speed_per_s: np.ndarray
    steady_s: np.ndarray
    state: np.ndarray
    pwm: np.ndarray


def corner_loads(total_n: float, center_xy: np.ndarray) -> np.ndarray:
    """Distribute a vertical load among four corners with bilinear weights."""
    weights = np.prod((1.0 + center_xy[:, None, :] * SENSOR_XY) / 2.0, axis=2)
    return total_n * weights


def pressure_center(loads_n: np.ndarray) -> np.ndarray:
    """Recover the normalized center of pressure where the seat is occupied."""
    total_n = loads_n.sum(axis=1)
    seated = total_n >= SEATED_LOAD_N
    center_xy = np.full((len(total_n), 2), np.nan)
    weighted_sum = (loads_n[seated, :, None] * SENSOR_XY).sum(axis=1)
    center_xy[seated] = weighted_sum / total_n[seated, None]
    return center_xy


def make_scenario() -> Scenario:
    """Create two sittings with a brief movement and a sustained lean."""
    neutral = (0.08, -0.06)
    brief = (0.57, -0.03)
    lean = (0.54, 0.02)
    second_neutral = (-0.18, 0.12)

    # Each segment gives duration, phase, load, and beginning/ending CoP.
    segments = (
        (3, "empty", 0.0, neutral, neutral),
        (5, "initial_calibration", 650.0, neutral, neutral),
        (4, "neutral", 650.0, neutral, neutral),
        (1, "brief_motion", 650.0, neutral, brief),
        (1, "brief_lean", 650.0, brief, brief),
        (1, "brief_return", 650.0, brief, neutral),
        (5, "neutral", 650.0, neutral, neutral),
        (2, "lean_transition", 650.0, neutral, lean),
        (22, "sustained_lean", 650.0, lean, lean),
        (2, "recovery", 650.0, lean, neutral),
        (4, "recovered", 650.0, neutral, neutral),
        (4, "empty", 0.0, neutral, neutral),
        (5, "reseat_calibration", 520.0, second_neutral, second_neutral),
        (5, "reseated", 520.0, second_neutral, second_neutral),
    )
    phases = []
    true_centers = []
    loads = []
    for duration_s, name, load_n, start_xy, end_xy in segments:
        count = duration_s * SAMPLE_RATE_HZ
        path_xy = np.linspace(start_xy, end_xy, count, endpoint=False)
        phases.extend([name] * count)
        loads.append(corner_loads(load_n, path_xy))
        true_centers.append(path_xy if load_n else np.full_like(path_xy, np.nan))

    loads_n = np.concatenate(loads)
    noise_n = np.random.default_rng(RANDOM_SEED).normal(0.0, NOISE_STD_N, loads_n.shape)
    return Scenario(
        time_s=np.arange(len(phases)) / SAMPLE_RATE_HZ,
        phase=np.array(phases),
        true_xy=np.concatenate(true_centers),
        loads_n=np.maximum(loads_n + noise_n, 0.0),
    )


def analyze_posture(loads_n: np.ndarray) -> Posture:
    """Apply per-sitting calibration and drive a virtual haptic output."""
    centers_xy = pressure_center(loads_n)
    count = len(centers_xy)
    baseline_xy = np.full((count, 2), np.nan)
    relative_xy = np.full((count, 2), np.nan)
    speed_per_s = np.full(count, np.nan)
    steady_s = np.zeros(count)
    state = np.full(count, "vacant", dtype="<U11")
    pwm = np.zeros(count, dtype=int)

    tare_samples = []
    tare_xy = filtered_xy = None
    steady_start = None

    for index, center_xy in enumerate(centers_xy):
        if np.isnan(center_xy).any():
            tare_samples.clear()
            tare_xy = filtered_xy = None
            steady_start = None
            continue

        if len(tare_samples) < TARE_FRAMES:
            tare_samples.append(center_xy)
            state[index] = "calibrating"
            continue

        if tare_xy is None:
            tare_xy = np.mean(tare_samples, axis=0)
            filtered_xy = tare_xy

        change_xy = EMA_ALPHA * (center_xy - filtered_xy)
        filtered_xy = filtered_xy + change_xy
        baseline_xy[index] = tare_xy
        relative_xy[index] = filtered_xy - tare_xy
        speed_per_s[index] = np.linalg.norm(change_xy) * SAMPLE_RATE_HZ

        if np.linalg.norm(relative_xy[index]) <= DISTANCE_LIMIT:
            state[index] = "balanced"
            steady_start = None
        elif speed_per_s[index] > SPEED_LIMIT_PER_S:
            state[index] = "moving"
            steady_start = None
        else:
            if steady_start is None:
                steady_start = index
            elapsed_frames = index - steady_start
            steady_s[index] = elapsed_frames / SAMPLE_RATE_HZ
            if elapsed_frames < HOLD_FRAMES:
                state[index] = "watching"
                continue

            state[index] = "alert"
            ramp_fraction = min(1.0, (elapsed_frames - HOLD_FRAMES) / RAMP_FRAMES)
            pwm[index] = round(PWM_MIN + (PWM_MAX - PWM_MIN) * ramp_fraction)

    return Posture(baseline_xy, relative_xy, speed_per_s, steady_s, state, pwm)


def main() -> None:
    scenario = make_scenario()
    centers_xy = pressure_center(scenario.loads_n)
    seated = np.isfinite(centers_xy[:, 0])
    error = np.linalg.norm(centers_xy[seated] - scenario.true_xy[seated], axis=1)
    print(f"Samples: {len(seated)}, seated: {np.count_nonzero(seated)}")
    print(f"Center-of-pressure RMS error: {np.sqrt(np.mean(error**2)):.5f}")
    posture = analyze_posture(scenario.loads_n)
    alerts = np.flatnonzero(posture.state == "alert")
    if alerts.size:
        first, last = alerts[0], alerts[-1]
        print(f"Alert: {scenario.time_s[first]:.2f}–{scenario.time_s[last]:.2f} s")
    print(f"Maximum virtual PWM: {posture.pwm.max()}")


if __name__ == "__main__":
    main()
