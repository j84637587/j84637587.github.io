import matplotlib.pyplot as plt


# 可用套件 sklearn.metrics 的 auc
def auc(fpr, tpr):
    auc_value = 0.0
    for i in range(1, len(fpr)):
        auc_value += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc_value

# 可用套件 sklearn.metrics 的 roc_curve
def roc_curve(y_true, y_score):
    sorted_indices = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    y_true_sorted = [y_true[i] for i in sorted_indices]
    y_score_sorted = [y_score[i] for i in sorted_indices]

    thresholds = sorted(set(y_score_sorted), reverse=True)
    tpr = []
    fpr = []

    # 計算每個閾值下的 TPR 和 FPR
    for threshold in thresholds:
        # 根據閾值預測標籤
        y_pred = [1 if score >= threshold else 0 for score in y_score_sorted]

        # 計算TPR和FPR
        tp = sum(1 for i in range(len(y_true_sorted)) if y_true_sorted[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true_sorted)) if y_true_sorted[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_true_sorted)) if y_true_sorted[i] == 1 and y_pred[i] == 0)
        tn = sum(1 for i in range(len(y_true_sorted)) if y_true_sorted[i] == 0 and y_pred[i] == 0)

        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return fpr, tpr, thresholds

def main():
    # 實際標籤和模型輸出機率
    y_true = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.6, 0.1, 0.4, 0.7, 0.8, 0.7, 0.6, 0.9]

    # 計算 ROC 曲線
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # 畫圖
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')

    for x, y, t in zip(fpr, tpr, thresholds):
        print(f'Threshold: {t:.2f}, FPR: {x:.2f}, TPR: {y:.2f}')
        plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10, ha='right')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()