import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Configuration
N_PARTICLES = 40
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def superimpose_coordinates(
    reference_coords: np.ndarray, moving_coords: np.ndarray
) -> np.ndarray:
    """
    Superimposes a set of moving coordinates onto a reference set using the
    Kabsch algorithm.

    This involves centering both sets of coordinates, calculating the optimal
    rotation matrix via Singular Value Decomposition (SVD), and then applying
    the rotation and translation to the moving coordinates.

    Args:
        reference_coords: A (N, 2) numpy array of reference particle coordinates.
        moving_coords: A (N, 2) numpy array of coordinates to be aligned.

    Returns:
        A (N, 2) numpy array of the transformed (aligned) coordinates.
    """
    # 1. Center the coordinates by subtracting their respective centroids.
    ref_center = np.mean(reference_coords, axis=0)
    mov_center = np.mean(moving_coords, axis=0)
    ref_centered = reference_coords - ref_center
    mov_centered = moving_coords - mov_center

    # 2. Compute the covariance matrix.
    covariance_matrix = np.dot(mov_centered.T, ref_centered)

    # 3. Use Singular Value Decomposition (SVD) to find the optimal rotation.
    u, _, vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(u, vt)

    # 4. Correct for reflection. If the determinant of the rotation matrix is -1,
    # it's a reflection, not a proper rotation. We fix this by flipping the sign
    # of the last column of U, which corresponds to the smallest singular value.
    if np.linalg.det(rotation_matrix) < 0:
        vt[-1, :] *= -1
        rotation_matrix = np.dot(u, vt)

    # 5. Apply the rotation to the original (uncentered) moving coordinates
    # and add the translation vector to align the centroids.
    transformed_coords = np.dot(moving_coords, rotation_matrix) + (
        ref_center - np.dot(mov_center, rotation_matrix)
    )

    return transformed_coords


def calculate_mse(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between two sets of coordinates
    after superimposing them.

    Args:
        coords1: The reference coordinates.
        coords2: The moving coordinates to align and compare.

    Returns:
        The mean squared error between the aligned coordinate sets.
    """
    aligned_coords2 = superimpose_coordinates(coords1, coords2)
    error = (coords1 - aligned_coords2) ** 2
    return np.mean(error)


# Trajectory and State Generation

def create_mode_generator(
    n_particles: int = N_PARTICLES,
) -> Tuple[Callable, int]:
    """
    Creates a function that generates particle chains based on sine curves.

    The function defines several "modes" (stable states), each represented
    by a phase-shifted sine wave. It returns a callable that can generate
    a chain representing an interpolation between any two of these modes.

    Args:
        n_particles: The number of particles in the chain.

    Returns:
        A tuple containing:
        - A function that interpolates between modes.
        - The number of defined modes.
    """
    t = np.linspace(0, 1, n_particles)
    # Each mode is defined by a phase shift 'a' for sin(2*pi*t + a)
    modes = np.array([[0.0], [np.pi * 0.5], [np.pi]])

    def _create_curve(coeffs: np.ndarray) -> np.ndarray:
        """Generates a single curve from a coefficient vector."""
        (a,) = coeffs
        return np.stack([t, np.sin(2 * np.pi * t + a)], axis=-1)

    def _interpolate_and_center(
        mode1_idx: int, mode2_idx: int, progress: float
    ) -> np.ndarray:
        """
        Generates a particle chain by interpolating between two modes and
        centering the result.

        Args:
            mode1_idx: The index of the starting mode.
            mode2_idx: The index of the target mode.
            progress: The interpolation factor (0.0 to 1.0).

        Returns:
            A (n_particles, 2) numpy array of particle positions.
        """
        start_coeffs = modes[mode1_idx]
        end_coeffs = modes[mode2_idx]
        interp_coeffs = start_coeffs + (end_coeffs - start_coeffs) * progress
        positions = _create_curve(interp_coeffs)

        # Center the chain relative to the interpolated mean position
        start_mean = np.mean(_create_curve(modes[mode1_idx]), axis=0)
        end_mean = np.mean(_create_curve(modes[mode2_idx]), axis=0)
        interp_mean = start_mean + (end_mean - start_mean) * progress
        return positions - interp_mean

    return _interpolate_and_center, len(modes)


def generate_correlated_noise(
    prev_scales: np.ndarray, amplitude: float = 0.01, n_particles: int = N_PARTICLES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates smooth, time-correlated noise for the particle chain.

    The noise is a sum of sine waves with slowly changing amplitudes.

    Args:
        prev_scales: The scaling factors from the previous timestep.
        amplitude: The overall amplitude of the random walk for the scales.
        n_particles: The number of particles in the chain.

    Returns:
        A tuple containing:
        - The generated noise array (n_particles, 2).
        - The updated scaling factors.
    """
    # Update scales with a small random walk to ensure temporal correlation
    new_scales = prev_scales + np.random.randn(4, 2) * amplitude
    new_scales *= 0.5  # Damping factor

    # Create noise from a basis of sine functions
    ts = np.linspace(0, 1, n_particles)
    basis_functions = np.sin(np.arange(4)[:, None] * np.pi * ts[None, :])
    chain_noise = (basis_functions[:, :, None] * new_scales[:, None, :]).sum(0)

    return chain_noise, new_scales


class TrajectoryGenerator:
    """
    A generator that simulates a molecular trajectory using a Markov chain.

    The system transitions between different conformational states (modes)
    based on a transition probability matrix. The trajectory includes smooth
    interpolation between states, as well as random noise and rotation.
    """

    def __init__(
        self,
        mode_generator: Callable,
        transition_prob: np.ndarray,
        initial_probs: np.ndarray,
        noise_amplitude: float,
        transition_duration: int,
    ):
        """
        Initializes the simulator's state.
        """
        self.mode_generator = mode_generator
        self.transition_prob = transition_prob
        self.noise_amplitude = noise_amplitude
        self.transition_duration = transition_duration

        # State variables
        self.n_states = initial_probs.shape[0]
        self.state_indices = np.arange(self.n_states)
        self.current_state = np.random.choice(self.state_indices, p=initial_probs)
        self.previous_state = self.current_state
        self.transition_counter = 0  # Time remaining in current transition
        self.noise_scales = np.zeros((4, 2))
        self.global_rotation = 0.0
        self.global_translation = np.zeros(2)
        self.base_positions = self.mode_generator(self.current_state, self.current_state, 0.0)

    def step(self) -> Tuple[np.ndarray, Dict]:
        """
        Generates the next frame in the trajectory.

        Returns:
            A tuple containing:
            - The (N, 2) array of particle positions for the current frame.
            - A dictionary of the current simulation state.
        """
        # State Transition Logic
        if self.transition_counter == 0:
            # Not in a transition, so decide the next state
            new_state = np.random.choice(
                self.state_indices, p=self.transition_prob[self.current_state]
            )
            if new_state != self.current_state:
                # Start a new transition
                self.previous_state = self.current_state
                self.current_state = new_state
                self.transition_counter = self.transition_duration
        else:
            # Interpolation Logic
            self.transition_counter -= 1
            progress = 1.0 - (self.transition_counter / self.transition_duration)
            self.base_positions = self.mode_generator(
                self.previous_state, self.current_state, progress
            )

        # Noise and Transformation Logic
        # 1. Generate correlated noise
        chain_noise, self.noise_scales = generate_correlated_noise(
            self.noise_scales
        )

        # 2. Apply random rotation and translation to the whole chain
        center = np.mean(self.base_positions, axis=0)
        centered_pos = self.base_positions - center

        # Update global rotation and translation with a small random walk
        self.global_rotation += (np.random.rand() - 0.5) * 0.02
        self.global_translation += np.random.randn(2) * 0.02

        rot_matrix = np.array(
            [
                [np.cos(self.global_rotation), -np.sin(self.global_rotation)],
                [np.sin(self.global_rotation), np.cos(self.global_rotation)],
            ]
        )
        rotated_pos = centered_pos.dot(rot_matrix)

        # 3. Combine base positions, noise, and transformations
        final_positions = (
            rotated_pos
            + (center + self.global_translation)
            + chain_noise * self.noise_amplitude
        )

        current_args = {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "transition_counter": self.transition_counter,
            "noise_scales_mean_abs": np.abs(self.noise_scales).mean(),
        }

        return final_positions, current_args


def setup_trajectory_generator(
    noise_amplitude: float, transition_duration: int, escape_time: int
) -> TrajectoryGenerator:
    """
    Configures and initializes the TrajectoryGenerator.

    Args:
        noise_amplitude: The amplitude of the correlated noise.
        transition_duration: The number of frames for a state transition.
        escape_time: The average number of frames before a state transition.

    Returns:
        An initialized TrajectoryGenerator instance.
    """
    mode_generator, n_modes = create_mode_generator(n_particles=N_PARTICLES)

    # Calculate the probability of staying in the same state per frame
    # to achieve the desired average escape_time.
    # P(stay)^T = 0.5  =>  T*log(P(stay)) = log(0.5)
    retain_prob = np.exp(np.log(0.5) / escape_time)
    
    # Define the transition matrix for a 3-state system (0 -> 1 -> 2 -> 0)
    initial_probs = np.array([1.0, 0.0, 0.0])
    transition_prob = np.array(
        [
            [retain_prob, 1 - retain_prob, 0],
            [0, retain_prob, 1 - retain_prob],
            [1 - retain_prob, 0, retain_prob],
        ]
    )
    assert np.allclose(transition_prob.sum(axis=-1), 1.0)

    return TrajectoryGenerator(
        mode_generator=mode_generator,
        transition_prob=transition_prob,
        initial_probs=initial_probs,
        noise_amplitude=noise_amplitude,
        transition_duration=transition_duration,
    )


# Main Execution and Visualization

def main():
    """
    Main function to run the simulation, generate trajectory data,
    and render it as a video.
    """
    # Simulation Parameters
    TOTAL_STEPS = 1000
    TRANSITION_DURATION = 30  # Frames
    ESCAPE_TIME = 300         # Average frames before transition
    NOISE_AMPLITUDE = 0.01
    FRAME_SKIP = 1
    OUTPUT_DIR = Path("frames")
    OUTPUT_VIDEO = "out.mp4"

    # 1. Setup and Run Simulation
    generator = setup_trajectory_generator(
        noise_amplitude=NOISE_AMPLITUDE,
        transition_duration=TRANSITION_DURATION,
        escape_time=ESCAPE_TIME,
    )

    # Initialize storage for trajectory data
    initial_pos, initial_args = generator.step()
    trajectory = np.empty((TOTAL_STEPS,) + initial_pos.shape)
    states = np.empty(TOTAL_STEPS, dtype=np.int32)
    prev_states = np.empty(TOTAL_STEPS, dtype=np.int32)

    trajectory[0] = initial_pos
    states[0] = initial_args["state"]
    prev_states[0] = initial_args["previous_state"]

    print("Generating trajectory data...")
    pbar = tqdm(range(1, TOTAL_STEPS))
    for t in pbar:
        sample, args = generator.step()
        trajectory[t] = sample
        states[t] = args["state"]
        prev_states[t] = args["previous_state"]
        pbar.set_description(
            f"State: {args['state']} | "
            f"Transitioning: {args['transition_counter'] > 0} | "
            f"Noise: {args['noise_scales_mean_abs']:.4f}"
        )

    # 2. Prepare for Rendering (Color Interpolation)
    print("Calculating frame colors for transitions...")
    # This logic creates a smooth color transition between states in the video.
    # `is_transitioning` is True for frames that are part of a state change.
    is_transitioning = states != prev_states
    
    # `transition_starts` marks the first frame of each new transition.
    transition_starts = is_transitioning & ~np.roll(is_transitioning, 1)
    
    # `progress_in_transition` will hold a value from 0.0 to 1.0 for each frame.
    progress_in_transition = np.zeros(TOTAL_STEPS)
    
    # Iterate through the trajectory and calculate the progress for each transition.
    current_transition_frame = 0
    for i in range(TOTAL_STEPS):
        if transition_starts[i]:
            current_transition_frame = 1
        elif is_transitioning[i]:
            current_transition_frame += 1
        else:
            current_transition_frame = 0
        
        if current_transition_frame > 0:
            progress_in_transition[i] = min(1.0, current_transition_frame / TRANSITION_DURATION)

    # Define base colors for each state (e.g., Red, Green, Blue)
    state_colors = np.eye(3)
    frame_colors = np.zeros((TOTAL_STEPS, 3))

    for i in range(TOTAL_STEPS):
        start_color = state_colors[prev_states[i]]
        end_color = state_colors[states[i]]
        frame_colors[i] = start_color + (end_color - start_color) * progress_in_transition[i]


    # 3. Render Frames and Create Video
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    print(f"Rendering {TOTAL_STEPS // FRAME_SKIP} frames to '{OUTPUT_DIR}'...")
    mins = trajectory.min(axis=(0, 1))
    maxs = trajectory.max(axis=(0, 1))

    for i in tqdm(range(0, TOTAL_STEPS, FRAME_SKIP)):
        plt.figure(figsize=(6, 6))
        plt.plot(*trajectory[i].T, lw=2.0, c=frame_colors[i])
        plt.xlim(mins[0] - 0.1, maxs[0] + 0.1)
        plt.ylim(mins[1] - 0.1, maxs[1] + 0.1)
        plt.axis("off")
        plt.savefig(OUTPUT_DIR / f"{i:04d}.png", bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print(f"Compiling video '{OUTPUT_VIDEO}' with ffmpeg...")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-framerate", "30",
                "-pattern_type", "glob",
                "-i", str(OUTPUT_DIR / "*.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y", # Overwrite output file if it exists
                OUTPUT_VIDEO,
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("Video compilation successful.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n--- FFMPEG Error")
        print("Could not create video. Please ensure ffmpeg is installed and in your system's PATH.")
        if isinstance(e, subprocess.CalledProcessError):
            print("FFMPEG stdout:", e.stdout)
            print("FFMPEG stderr:", e.stderr)
        return
    finally:
        # Clean up frames directory
        print(f"Cleaning up '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
        print("Done.")


if __name__ == "__main__":
    main()
