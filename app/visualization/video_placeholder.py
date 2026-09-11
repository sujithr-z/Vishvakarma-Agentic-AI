"""
Video / Simulation Panel & Live Video Player (Figure 3).
Loads and renders frames from 'video_plot/model_3d.mp4' with real-time looping
animation inside the Matplotlib interactive window and static fallback for image export.
"""
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _resolve_video_path(video_path: Optional[str]) -> Path:
    """Resolve video path relative to workspace or absolute."""
    resolved = Path(video_path) if video_path else Path("video_plot/model_3d.mp4")
    if not resolved.is_absolute():
        if not resolved.exists():
            workspace_root = Path(__file__).resolve().parent.parent.parent
            resolved = workspace_root / (video_path or "video_plot/model_3d.mp4")
    return resolved


def load_video_sample_frames(video_path: Optional[str] = "video_plot/model_3d.mp4", max_frames: int = 180, stride: int = 2) -> List[np.ndarray]:
    """
    Extract video frames for smooth looping animation in Matplotlib.
    
    Args:
        video_path: Path to MP4 file.
        max_frames: Maximum number of frames to cache in memory.
        stride: Frame skip stride to maintain high performance.
        
    Returns:
        List of RGB numpy image arrays.
    """
    if not HAS_CV2:
        return []

    p = _resolve_video_path(video_path)
    if not p.exists():
        return []

    frames = []
    try:
        cap = cv2.VideoCapture(str(p))
        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_idx += 1
        cap.release()
    except Exception:
        pass

    return frames


def render_video_placeholder(
    ax: plt.Axes,
    video_path: Optional[str] = "video_plot/model_3d.mp4"
) -> Any:
    """
    Render Figure 3: Simulation Video Panel into the provided Matplotlib axis.
    Displays the first real video frame or aesthetic simulation poster card.
    
    Args:
        ax: Matplotlib Axis to render on.
        video_path: Path to the 3D model / walkthrough video file.
        
    Returns:
        The Matplotlib image artist if frame was loaded, or None.
    """
    ax.clear()
    p = _resolve_video_path(video_path)
    has_video = p.exists()
    size_mb = f"{p.stat().st_size / (1024 * 1024):.1f} MB" if has_video else "N/A"
    filename = p.name if has_video else "model_3d.mp4"

    # Attempt to load first frame as poster image
    poster_loaded = False
    im_artist = None

    if has_video and HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(p))
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_artist = ax.imshow(rgb)
                ax.set_title("Figure 3: 3D Shelter Simulation (model_3d.mp4)", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
                ax.axis("off")
                poster_loaded = True
        except Exception:
            poster_loaded = False

    if not poster_loaded:
        ax.set_facecolor("#0f172a")
        ax.set_title("Figure 3: 3D Shelter Simulation (model_3d.mp4)", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Player frame
        player_box = patches.FancyBboxPatch(
            (0.5, 0.5), 9.0, 9.0,
            boxstyle="round,pad=0.2,rounding_size=0.4",
            facecolor="#0f172a",
            edgecolor="#38bdf8",
            linewidth=2.0,
            zorder=1
        )
        ax.add_patch(player_box)

        # Center play circle
        center_circle = plt.Circle((5.0, 5.6), 1.3, facecolor="#0284c7", edgecolor="#38bdf8", linewidth=1.5, zorder=3)
        ax.add_patch(center_circle)

        # Triangle play symbol
        triangle = patches.Polygon(
            [[4.6, 4.8], [4.6, 6.4], [5.8, 5.6]],
            closed=True,
            facecolor="#ffffff",
            edgecolor="#ffffff",
            zorder=4
        )
        ax.add_patch(triangle)

        badge_color = "#10b981" if has_video else "#f59e0b"
        badge_text = "● 3D VIDEO SIMULATION LOADED" if has_video else "○ SIMULATION QUEUED"
        ax.text(5.0, 3.4, badge_text, fontsize=9, fontweight="bold", color=badge_color, ha="center", va="center", zorder=5)

        status_desc = (
            f"File: video_plot/{filename} ({size_mb})\n"
            "Model: 3D Modular Structure & Airflow Dynamics\n"
            "Status: Interactive Video Player Ready"
        )
        ax.text(5.0, 1.8, status_desc, fontsize=8.5, color="#94a3b8", ha="center", va="center", fontfamily="sans-serif", zorder=5)

    return im_artist


def attach_video_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    video_path: Optional[str] = "video_plot/model_3d.mp4",
    interval_ms: int = 40
) -> Optional[animation.FuncAnimation]:
    """
    Attach real-time video frame animation to Figure 3 axis for live interactive playback.
    
    Args:
        fig: Matplotlib Figure.
        ax: Matplotlib Axis for Figure 3.
        video_path: Path to MP4 video.
        interval_ms: Milliseconds per frame (~25 FPS).
        
    Returns:
        FuncAnimation instance (must be kept in scope to animate).
    """
    frames = load_video_sample_frames(video_path, max_frames=200, stride=2)
    if not frames:
        return None

    ax.clear()
    im = ax.imshow(frames[0])
    ax.set_title("Figure 3: Live 3D Simulation Walkthrough (model_3d.mp4)", fontsize=11, fontweight="bold", color="#1e293b", pad=8)
    ax.axis("off")

    num_frames = len(frames)

    def update_frame(i):
        frame_idx = i % num_frames
        im.set_data(frames[frame_idx])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=num_frames,
        interval=interval_ms,
        blit=True,
        cache_frame_data=False
    )
    return anim
