
#!/usr/bin/env python3
"""FEATS live force feedback and latency measurements."""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.animation import FuncAnimation

# FEATS imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataloader import normalize, unnormalize
from models.unet import UNet


# ==========================================
# 1. GelSight Mini image capture
# ==========================================
def capture_image(cam, imgw=320, imgh=240):

    for _ in range(5):
        ret, frame = cam.read()

        if ret and frame is not None and frame.size > 0:
            break
    else:
        raise RuntimeError("GelSight Mini image read failed")

    # Original FEATS preprocessing
    img = cv2.resize(frame, (895, 672))

    bx = int(img.shape[0] / 7)
    by = int(np.floor(img.shape[1] / 7))

    img = img[
        bx:img.shape[0] - bx,
        by:img.shape[1] - by
    ]

    img = cv2.resize(img[:, :-1], (imgw, imgh))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================
# 2. FEATS force prediction
# ==========================================
def make_prediction(img, model, device, config):

    data = normalize(
        {"gs_img": img},
        config["norm_file"]
    )

    if config["calibration_file"] is not None:

        calibration = np.load(config["calibration_file"])

        M = np.float32([
            [1, 0, calibration[0]],
            [0, 1, calibration[1]]
        ])

        warped = cv2.warpAffine(
            data["gs_img"],
            M,
            (320, 240),
            borderMode=cv2.BORDER_REPLICATE
        )

        inputs = (
            torch.from_numpy(warped)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

    else:

        inputs = (
            torch.from_numpy(data["gs_img"])
            .float()
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .to(device)
        )

    with torch.inference_mode():

        outputs = model(inputs)

        output_grid = outputs.squeeze(0).permute(1, 2, 0)

        grids = [
            unnormalize(
                output_grid[:, :, i],
                key,
                config["norm_file"]
            )
            for i, key in enumerate(
                ("grid_x", "grid_y", "grid_z")
            )
        ]

    return tuple(
        grid.detach().cpu().numpy()
        for grid in grids
    )


# ==========================================
# 3. Force feedback decision
# ==========================================
def calculate_feedback(grid_z, target, tolerance):

    measured = abs(float(np.sum(grid_z)))

    difference = measured - target

    if difference > tolerance:

        status = "TOO HIGH"
        action = "DECREASE FORCE"
        color = "firebrick"

    elif difference < -tolerance:

        status = "TOO LOW"
        action = "INCREASE FORCE"
        color = "darkorange"

    else:

        status = "OK"
        action = "WITHIN TOLERANCE"
        color = "forestgreen"

    return measured, difference, status, action, color


def update_feedback_text(
    feedback_text, target, tolerance, result
):

    measured, difference, status, action, color = result

    feedback_text.set_text(
        f"Target: {target:.2f} N    "
        f"Measured |Fz|: {measured:.2f} N\n"
        f"Difference: {difference:+.2f} N    "
        f"Status: {status}\n"
        f"Action: {action}    "
        f"Tolerance: +/-{tolerance:.2f} N"
    )

    feedback_text.set_color(color)


# ==========================================
# 4. Live prediction and timing
# ==========================================
def animate(
    _frame, cam, model, device, config,
    ims, axs, target, tolerance,
    feedback_text, speed_text, timing
):

    # Start image acquisition timing
    t0 = time.perf_counter()

    try:
        gs_img = capture_image(cam)

    except RuntimeError as exc:
        print(f"WARNING: {exc}", flush=True)
        return

    t1 = time.perf_counter()

    # FEATS inference
    px, py, pz = make_prediction(
        gs_img, model, device, config
    )

    # Synchronize GPU if applicable
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t2 = time.perf_counter()

    # Force feedback decision only
    feedback_result = calculate_feedback(
        pz, target, tolerance
    )

    t3 = time.perf_counter()

    # Measured processing times
    capture_ms = (t1 - t0) * 1000
    inference_ms = (t2 - t1) * 1000
    feedback_ms = (t3 - t2) * 1000

    force_ready_ms = (t2 - t0) * 1000
    decision_ready_ms = (t3 - t0) * 1000

    # Measure actual force result update intervals
    last = timing["last_time"]

    if last is not None:
        gap_ms = (t3 - last) * 1000
        timing["intervals"].append(gap_ms)

    timing["last_time"] = t3

    intervals = timing["intervals"]

    if intervals:

        avg_gap_ms = sum(intervals) / len(intervals)
        actual_hz = 1000 / avg_gap_ms
        longest_ms = max(intervals)

        rate_label = (
            f"Actual update: {actual_hz:.2f} Hz  |  "
            f"Avg gap: {avg_gap_ms:.1f} ms  |  "
            f"Longest gap: {longest_ms:.1f} ms"
        )

    else:

        rate_label = "Actual update: measuring..."

    # Update feedback and timing text
    update_feedback_text(
        feedback_text,
        target,
        tolerance,
        feedback_result
    )

    speed_text.set_text(
        f"Capture: {capture_ms:.1f} ms   |   "
        f"FEATS: {inference_ms:.1f} ms   |   "
        f"Feedback: {feedback_ms:.3f} ms\n"
        f"Capture-to-force: {force_ready_ms:.1f} ms   |   "
        f"Capture-to-decision: {decision_ready_ms:.1f} ms\n"
        f"{rate_label}"
    )

    # Update FEATS images
    for im, grid in zip(
        ims[:3], (px, py, pz)
    ):
        im.set_data(grid)

    ims[3].set_data(gs_img.astype(np.uint8))

    for ax, name, grid in zip(
        axs[:3],
        ("x", "y", "z"),
        (px, py, pz)
    ):
        ax.set_title(
            f"{name}-force: {np.sum(grid):.2f} N",
            fontsize=12
        )

    # Terminal results
    print(
        f"Capture: {capture_ms:.1f} ms | "
        f"FEATS: {inference_ms:.1f} ms | "
        f"Feedback: {feedback_ms:.3f} ms | "
        f"Force ready: {force_ready_ms:.1f} ms | "
        f"Decision ready: {decision_ready_ms:.1f} ms | "
        f"{rate_label}",
        flush=True
    )


# ==========================================
# 5. Main program
# ==========================================
def main(config, target, tolerance, camera_index=0):

    cam = cv2.VideoCapture(camera_index)

    if not cam.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}"
        )

    try:

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")

        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        print(f"FEATS device: {device}", flush=True)

        # Load pretrained FEATS model
        model = UNet(
            enc_chs=config["enc_chs"],
            dec_chs=config["dec_chs"],
            out_sz=config["output_size"]
        )

        model.load_state_dict(
            torch.load(
                config["model"],
                map_location="cpu",
                weights_only=True
            )
        )

        model.eval().to(device)

        print("FEATS model loaded", flush=True)

        # Initial prediction
        image = capture_image(cam)

        grids = make_prediction(
            image, model, device, config
        )

        # Create four panels
        fig, axs = plt.subplots(
            1, 4, figsize=(16, 5)
        )

        fig.canvas.manager.set_window_title(
            "FEATS Force Feedback Latency"
        )

        fig.suptitle(
            "FEATS LIVE FORCE FEEDBACK",
            y=0.98,
            fontsize=16
        )

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.67,
            bottom=0.18,
            wspace=0.38
        )

        feedback_text = fig.text(
            0.5, 0.88, "",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            linespacing=1.5
        )

        speed_text = fig.text(
            0.5, 0.76,
            "Measuring latency...",
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.5
        )

        initial_feedback = calculate_feedback(
            grids[2], target, tolerance
        )

        update_feedback_text(
            feedback_text,
            target,
            tolerance,
            initial_feedback
        )

        limits = [
            (-0.029, 0.029),
            (-0.029, 0.029),
            (-0.17, 0.0)
        ]

        ims = []

        for ax, name, grid, (vmin, vmax) in zip(
            axs[:3],
            ("x", "y", "z"),
            grids,
            limits
        ):

            im = ax.imshow(
                grid,
                origin="upper",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(
                f"{name}-force: {np.sum(grid):.2f} N",
                fontsize=12
            )

            ax.axis("off")

            fig.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                fraction=0.046,
                pad=0.04
            )

            ims.append(im)

        im_gs = axs[3].imshow(
            image.astype(np.uint8)
        )

        axs[3].set_title(
            "GelSight Mini Image",
            fontsize=12
        )

        axs[3].axis("off")

        ims.append(im_gs)

        fake_cbar = fig.colorbar(
            im_gs,
            ax=axs[3],
            orientation="horizontal",
            fraction=0.046,
            pad=0.04
        )

        fake_cbar.ax.set_visible(False)

        timing = {
            "last_time": None,
            "intervals": deque(maxlen=30)
        }

        ani = FuncAnimation(
            fig,
            animate,
            interval=100,
            blit=False,
            fargs=(
                cam,
                model,
                device,
                config,
                ims,
                axs,
                target,
                tolerance,
                feedback_text,
                speed_text,
                timing
            ),
            cache_frame_data=False
        )

        print("Live prediction started", flush=True)

        plt.show()

        _ = ani

    finally:

        cam.release()
        cv2.destroyAllWindows()


# ==========================================
# 6. Program entry point
# ==========================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FEATS force feedback latency"
    )

    parser.add_argument(
        "-c", "--config",
        default="predict_config.yaml"
    )

    parser.add_argument(
        "--target",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.target < 0 or args.tolerance < 0:
        parser.error(
            "Target and tolerance must be non-negative"
        )

    with open(
        args.config, "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    main(
        config,
        args.target,
        args.tolerance,
        args.camera
    )
