"""
Architecture Image Visualization Panel (Figure 2).
Loads and displays 'img_plot/arch_plan.jpeg' with graceful fallback to
a schematic specification card if missing or unreadable.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def render_architecture_panel(
    ax: plt.Axes,
    image_path: Optional[str] = "img_plot/arch_plan.jpeg",
    design_spec: Optional[Dict[str, Any]] = None
) -> None:
    """
    Render Figure 2: Shelter Architecture into the provided Matplotlib axis.
    
    Args:
        ax: Matplotlib Axis to render on.
        image_path: Path to the architectural blueprint/drawing image.
        design_spec: Optional shelter design dictionary for metadata annotations.
    """
    ax.clear()
    img_loaded = False

    resolved_path = Path(image_path) if image_path else Path("img_plot/arch_plan.jpeg")
    if not resolved_path.is_absolute():
        # Check relative to repo root
        if not resolved_path.exists():
            workspace_root = Path(__file__).resolve().parent.parent.parent
            resolved_path = workspace_root / image_path

    if resolved_path.exists():
        try:
            img = mpimg.imread(str(resolved_path))
            ax.imshow(img)
            ax.set_title("Shelter Architecture", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
            ax.axis("off")
            img_loaded = True
        except Exception:
            img_loaded = False

    if not img_loaded:
        # Fallback: Clean architectural schematic card
        ax.set_facecolor("#f8fafc")
        ax.set_title("Shelter Architecture (Schematic Blueprint)", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Draw decorative blueprint border
        rect = plt.Rectangle((0.5, 0.5), 9.0, 9.0, fill=True, facecolor="#f1f5f9", edgecolor="#3b82f6", linewidth=2.0, linestyle="--")
        ax.add_patch(rect)

        # Extract specifications
        geom = (design_spec or {}).get("geometry", {})
        roof = (design_spec or {}).get("roof", {})
        walls = (design_spec or {}).get("walls", {})

        length = geom.get("length", 6.0)
        width = geom.get("width", 4.0)
        height = geom.get("height", 3.0)
        opening_ratio = walls.get("opening_ratio", 0.25)
        overhang = roof.get("overhang", 0.60)
        mat_roof = roof.get("material", "Bamboo Mat Board / Thatch")
        mat_wall = walls.get("material", "Woven Bamboo Panels")

        info_text = (
            "📐 ARCHITECTURAL BLUEPRINT SPECIFICATION\n"
            "─────────────────────────────────────────\n"
            f"• Dimensions (L × W × H) : {length}m × {width}m × {height}m\n"
            f"• Floor Footprint Area  : {length * width:.1f} m²\n"
            f"• Operable Window Area   : {opening_ratio * 100:.0f}% of wall area\n"
            f"• Roof Overhang (Eaves)  : {overhang} m\n"
            f"• Roof Construction     : {mat_roof}\n"
            f"• Wall Construction     : {mat_wall}\n"
            "─────────────────────────────────────────\n"
            "Status: Blueprint image referenced at img_plot/arch_plan.jpeg"
        )
        ax.text(5.0, 5.0, info_text, fontsize=9.5, color="#1e293b", fontfamily="monospace", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2))
