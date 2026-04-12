import cv2
import numpy as np
import os

def calculate_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0
    
    return iou, dice

def evaluate_folder(pred_folder, gt_folder):
    if not os.path.exists(gt_folder):
        return None, None
        
    all_ious = []
    all_dices = []
    
    pred_files = [f for f in os.listdir(pred_folder) if f.startswith('res_')]
    
    for f in pred_files:
        gt_name = f.replace('res_', '')
        base_name = os.path.splitext(gt_name)[0]
        gt_path = os.path.join(gt_folder, base_name + ".png")
        
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_folder, base_name + ".jpg")

        if os.path.exists(gt_path):
            pred_img = cv2.imread(os.path.join(pred_folder, f), 0)
            gt_img = cv2.imread(gt_path, 0)
            
            if pred_img is not None and gt_img is not None:
                iou, dice = calculate_metrics(pred_img, gt_img)
                all_ious.append(iou)
                all_dices.append(dice)
            
    if not all_ious:
        return 0.0, 0.0
        
    return np.mean(all_ious), np.mean(all_dices)

if __name__ == "__main__":
    tasks = [
        ('TRAIN', 'results_train', 'train'),
        ('VALIDATION', 'results_validation', 'validation'),
        ('TEST', 'results_test', 'test')
    ]
    
    print("\n" + "="*50)
    print(f"{'Dataset Split':<15} | {'Mean IoU':<10} | {'Mean Dice':<10}")
    print("-" * 50)
    
    for name, pred, gt in tasks:
        miou, mdice = evaluate_folder(pred, gt)
        if miou is None:
            print(f"{name:<15}")
            continue
            
        print(f"{name:<15} | {miou:<10.4f} | {mdice:<10.4f}")


