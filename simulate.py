"""Simulate load sensing and haptic feedback for a seat cushion at 20 Hz."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Patch


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


def summarize(scenario: Scenario, posture: Posture) -> list[tuple[str, str, str]]:
    centers_xy = pressure_center(scenario.loads_n)
    seated = np.isfinite(centers_xy[:, 0])
    error = np.linalg.norm(centers_xy[seated] - scenario.true_xy[seated], axis=1)
    first, last = np.flatnonzero(posture.pwm)[[0, -1]]
    start = first - HOLD_FRAMES
    full_pwm = np.flatnonzero(posture.pwm == PWM_MAX)[0]
    initial = posture.baseline_xy[scenario.phase == "neutral"][0]
    reseated = posture.baseline_xy[scenario.phase == "reseated"][0]
    brief = np.char.startswith(scenario.phase, "brief")

    return [
        ("sample_rate", str(SAMPLE_RATE_HZ), "Hz"),
        ("sample_count", str(len(seated)), "samples"),
        ("seated_samples", str(np.count_nonzero(seated)), "samples"),
        ("cop_rms_error", f"{np.sqrt(np.mean(error**2)):.5f}", "normalized"),
        ("brief_motion_max_pwm", str(posture.pwm[brief].max()), "0-255"),
        ("empty_seat_max_pwm", str(posture.pwm[~seated].max()), "0-255"),
        ("steady_lean_started", f"{scenario.time_s[start]:.2f}", "s"),
        ("alert_started", f"{scenario.time_s[first]:.2f}", "s"),
        ("initial_pwm", str(posture.pwm[first]), "0-255"),
        ("full_pwm_reached", f"{scenario.time_s[full_pwm]:.2f}", "s"),
        ("max_pwm", str(posture.pwm.max()), "0-255"),
        ("pwm_off", f"{(last + 1) / SAMPLE_RATE_HZ:.2f}", "s"),
        ("initial_baseline_x", f"{initial[0]:.5f}", "normalized"),
        ("initial_baseline_y", f"{initial[1]:.5f}", "normalized"),
        ("reseated_baseline_x", f"{reseated[0]:.5f}", "normalized"),
        ("reseated_baseline_y", f"{reseated[1]:.5f}", "normalized"),
    ]


def plot_results(scenario: Scenario, posture: Posture, path: Path) -> None:
    fig = plt.figure(figsize=(12.4, 8.2), facecolor="#f5f7fb", layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 0.78, 0.9))
    loads_ax = fig.add_subplot(grid[0, :])
    trajectory_ax = fig.add_subplot(grid[1:, 0])
    distance_ax = fig.add_subplot(grid[1, 1], sharex=loads_ax)
    haptic_ax = fig.add_subplot(grid[2, 1], sharex=loads_ax)
    fig.suptitle("Smart posture cushion  |  Four-sensor simulation",
                 x=0.02, ha="left", fontsize=20, fontweight="bold", color="#172b45")

    for ax in (loads_ax, trajectory_ax, distance_ax, haptic_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#9aaabd")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    time_s = scenario.time_s
    end_s = len(time_s) / SAMPLE_RATE_HZ
    for index, (name, color) in enumerate((
        ("Rear left", "#386cb0"), ("Rear right", "#6e59a5"),
        ("Front left", "#008779"), ("Front right", "#c16420"),
    )):
        loads_ax.plot(time_s, scenario.loads_n[:, index], label=name, color=color)
    loads_ax.set(title="Four corner loads", ylabel="Load (N)", xlim=(0, end_s), ylim=(-10, 310))
    loads_ax.legend(loc="upper right", ncol=4, frameon=False)

    brief = np.char.startswith(scenario.phase, "brief")
    sustained = np.isin(scenario.phase, ("lean_transition", "sustained_lean", "recovery"))
    trajectory_ax.add_patch(Circle((0, 0), DISTANCE_LIMIT, linewidth=1.4,
                                   facecolor="#e3f5ee", edgecolor="#21836c",
                                   label="Allowed region"))
    # The phase masks keep disjoint sections from being joined by plot lines.
    for mask, label, color in (
        (brief, "Brief movement", "#386cb0"),
        (sustained, "Sustained lean and return", "#ce583a"),
    ):
        xy = np.where(mask[:, None], posture.relative_xy, np.nan)
        trajectory_ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.2, label=label)
    trajectory_ax.scatter(0, 0, color="#172b45", s=25, zorder=4, label="Personal baseline")
    trajectory_ax.set(title="Center-of-pressure path relative to baseline",
                      xlabel="Left (−) / Right (+)", ylabel="Rear (−) / Front (+)",
                      xlim=(-0.37, 0.60), ylim=(-0.38, 0.38), aspect="equal")
    trajectory_ax.legend(loc="lower left", framealpha=0.94, edgecolor="#e3e9ef", fontsize=8)

    distance = np.linalg.norm(posture.relative_xy, axis=1)
    distance_ax.plot(time_s, distance, color="#386cb0", linewidth=2)
    distance_ax.axhline(DISTANCE_LIMIT, color="#73859b", ls="--", label="Distance limit")
    distance_ax.set(title="Distance from personal baseline", ylabel="Normalized distance")
    distance_ax.legend(loc="upper right", frameon=False, fontsize=8)
    distance_ax.tick_params(labelbottom=False)

    state_colors = {
        "vacant": "#c8d1df", "calibrating": "#b9a6dd", "balanced": "#9bd5c0",
        "moving": "#99bae0", "watching": "#f2d58a", "alert": "#ee9a82",
    }
    change = np.flatnonzero(posture.state[1:] != posture.state[:-1]) + 1
    for start, stop in zip(np.r_[0, change], np.r_[change, len(time_s)], strict=True):
        haptic_ax.axvspan(start / SAMPLE_RATE_HZ, stop / SAMPLE_RATE_HZ,
                         color=state_colors[posture.state[start]], alpha=0.48, linewidth=0)
    haptic_ax.step(time_s, posture.pwm, where="post", color="#a13f2c", lw=2)
    haptic_ax.set(title="Detection state and virtual haptic output",
                  xlabel="Time (s)", ylabel="PWM (0–255)", ylim=(-6, 225))
    haptic_ax.legend(handles=[Patch(color=color, label=name.title(), alpha=0.65)
                              for name, color in state_colors.items()],
                     loc="upper left", ncol=3, fontsize=7.6, frameon=False)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = make_scenario()
    posture = analyze_posture(scenario.loads_n)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("metric", "value", "unit"))
        writer.writerows(summarize(scenario, posture))
    plot_results(scenario, posture, output_dir / "overview.png")
    print(f"Saved overview.png and summary.csv to {output_dir}")


if __name__ == "__main__":
    main()
