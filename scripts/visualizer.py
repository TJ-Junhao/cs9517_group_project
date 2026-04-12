import matplotlib.pyplot as plt
import numpy as np
import os


splits = ['TRAIN', 'VALIDATION', 'TEST']
iou_scores = [0.2574, 0.2980, 0.3144]
dice_scores = [0.3719, 0.3976, 0.4133]
baseline_iou = [0.22, 0.25, 0.28] 

# --- 2. 绘图设置 ---
x = np.arange(len(splits))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 IoU 和 Dice
rects1 = ax.bar(x - width/2, iou_scores, width, label='Mean IoU', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, dice_scores, width, label='Mean Dice', color='#ff7f0e', edgecolor='black')

ax.axhline(y=0.284, color='red', linestyle='--', alpha=0.6, label='ExG Baseline (Test)')


ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
ax.set_title('Performance Analysis across Dataset Splits', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.5)  
ax.legend(loc='upper left')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()


output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_results_chart.png')
plt.savefig(output_path, dpi=150)
plt.show()