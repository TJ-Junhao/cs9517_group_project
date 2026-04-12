import matplotlib.pyplot as plt
import numpy as np

# 1. 填入你跑出的数据
categories = ['Normal', 'Gaussian Noise\n(var=0.01)', 'Brightness Shift\n(beta=50)']
iou_scores = [0.4321, 0.3438, 0.4339]

# 2. 设置绘图样式
plt.figure(figsize=(10, 6))
colors = ['#4A90E2', '#D0021B', '#7ED321'] # 分别使用蓝、红、绿表示
bars = plt.bar(categories, iou_scores, color=colors, alpha=0.8, width=0.6)

# 3. 添加数值标签 (让图表一眼就能看出分数)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)

# 4. 添加基准线 (假设 ExG Baseline 是 0.38，如果没有可以去掉)
# plt.axhline(y=0.38, color='gray', linestyle='--', label='Baseline (ExG Only)')

# 5. 图表修饰
plt.title('Robustness Evaluation: Mean Shift + CRF Performance', fontsize=14, fontweight='bold')
plt.ylabel('Mean IoU Score', fontsize=12)
plt.ylim(0, 0.6) # 留出顶部空间
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 6. 保存图表
plt.savefig('robustness_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 对比图已生成并保存为 robustness_comparison.png")