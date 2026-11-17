import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

np.random.seed(42)
OUTPUT = r"C:\Users\HP\Desktop\DA\report_charts\04b_isolation_forest_concept.png"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
fig.suptitle("Isolation Forest: Algorithm Principle",
             fontsize=14, fontweight="bold", y=1.01)

# Left panel
ax1 = axes[0]
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.set_facecolor("#F8FBFF")
ax1.set_title("Normal Point — Requires Many Splits to Isolate",
              fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Feature 1 (e.g. Temperature)", fontsize=9)
ax1.set_ylabel("Feature 2 (e.g. Rainfall)", fontsize=9)

# Normal cluster
norm_x = np.random.normal(5, 1, 60)
norm_y = np.random.normal(5, 1, 60)
norm_x = np.clip(norm_x, 1, 9); norm_y = np.clip(norm_y, 1, 9)
ax1.scatter(norm_x, norm_y, c="#457B9D", s=40, alpha=0.6,
            zorder=3, label="Normal points")

# Target normal point (inside cluster)
ax1.scatter([5], [5], c="#E63946", s=120, zorder=5,
            marker="*", label="Target normal point")

# Many partition lines for a normal point
ax1.axhline(y=7.5, color="#2C3E50", linewidth=1.2, linestyle="--", alpha=0.7)
ax1.axvline(x=2.5, color="#2C3E50", linewidth=1.2, linestyle="--", alpha=0.7)
ax1.plot([2.5, 10], [3.5, 3.5], color="#2C3E50", linewidth=1.2,
         linestyle="--", alpha=0.7)
ax1.plot([4, 10], [6.5, 6.5], color="#2C3E50", linewidth=1.2,
         linestyle="--", alpha=0.7)
ax1.axvline(x=4, color="#2C3E50", linewidth=1.2, linestyle="--",
            alpha=0.7, ymin=0.35, ymax=0.65)
ax1.axvline(x=7, color="#2C3E50", linewidth=1.2, linestyle="--",
            alpha=0.7, ymin=0.35, ymax=0.65)

ax1.text(5, 0.4, "Path length = 6+ splits", ha="center",
         fontsize=10, fontweight="bold", color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", alpha=0.9))

split_patch = mpatches.Patch(color="#2C3E50", linestyle="--",
                              fill=False, label="Random split lines")
ax1.legend(handles=[
    mpatches.Patch(color="#457B9D", alpha=0.6, label="Normal points"),
    mpatches.Patch(color="#E63946", label="Target normal point (★)"),
    mpatches.Patch(facecolor="none", edgecolor="#2C3E50",
                   linestyle="--", label="Random splits"),
], fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3)

# Right panel: Anomaly
ax2 = axes[1]
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_facecolor("#FFF8F8")
ax2.set_title("Anomaly Point — Isolated in Few Splits",
              fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Feature 1 (e.g. Temperature)", fontsize=9)
ax2.set_ylabel("Feature 2 (e.g. Rainfall)", fontsize=9)

# Normal cluster
ax2.scatter(norm_x, norm_y, c="#457B9D", s=40, alpha=0.6,
            zorder=3, label="Normal points")

# Anomaly point (far from cluster — like 335mm rainfall day)
ax2.scatter([1.2], [9.1], c="#E63946", s=180, zorder=5,
            marker="X", label="Anomaly (e.g. 335mm flood day)")

# Very few partition lines needed
ax2.axhline(y=7.0, color="#E63946", linewidth=2, linestyle="--", alpha=0.8)
ax2.axvline(x=3.0, color="#E63946", linewidth=2, linestyle="--",
            alpha=0.8, ymin=0.7)

# Arrow pointing to anomaly
ax2.annotate("Anomaly isolated\nin just 2 splits!",
             xy=(1.2, 9.1), xytext=(3.5, 8.2),
             arrowprops=dict(arrowstyle="->", color="#E63946", lw=2),
             fontsize=9, fontweight="bold", color="#E63946",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#E63946", alpha=0.9))

ax2.text(5, 0.4, "Path length = 2 splits  →  ANOMALY",
         ha="center", fontsize=10, fontweight="bold", color="#E63946",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", alpha=0.9))

ax2.legend(handles=[
    mpatches.Patch(color="#457B9D", alpha=0.6, label="Normal points"),
    mpatches.Patch(color="#E63946", label="Anomaly point (✕)"),
    mpatches.Patch(facecolor="none", edgecolor="#E63946",
                   linestyle="--", label="Isolation splits"),
], fontsize=8, loc="center right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {OUTPUT}")
print("Caption: Fig. 4. Isolation Forest principle — normal points require")
print("many random splits to isolate (left) while anomalies are isolated")
print("in very few splits (right), making them detectable without labels.")