import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------
# Config (update here only)
# ---------------------------
ROW_START = 0          # Excel row 2
ROW_END   = 169        # Excel row 170 (inclusive in .loc)

COL_SLICE = slice(98, 118)

file_path = "Ex-Post-LLM Performance Analysis.xlsx" #download from Drive
manual_df = pd.read_excel(file_path, sheet_name="Joint manual labeling", engine="openpyxl")
pred_df   = pd.read_excel(file_path, sheet_name="LLM Original",          engine="openpyxl")

# Columns CU..DN
col_names = manual_df.columns[COL_SLICE]

# ---- Align rows across both sheets to avoid boolean index mismatches ----
desired_idx = manual_df.index.intersection(pred_df.index)
desired_idx = desired_idx[(desired_idx >= ROW_START) & (desired_idx <= ROW_END)]

manual_data = manual_df.loc[desired_idx, col_names].fillna(0).astype(int)
pred_data   = pred_df.loc[desired_idx, col_names].fillna(0).astype(int)

# Sanity check: make sure No_Biodiv exists in the selected columns
if "No_Biodiv" not in col_names:
    raise KeyError("Expected column 'No_Biodiv' not found in CU:DN. Please confirm column locations.")

# Prepare masks
no_biodiv_pred = pred_data["No_Biodiv"] == 1

# Build filtered vectors for overall metrics
filtered_true, filtered_pred = [], []
for col in col_names:
    if col == "No_Biodiv":
        filtered_true.append(manual_data[col].values)
        filtered_pred.append(pred_data[col].values)
    else:
        mask = ~no_biodiv_pred.values
        filtered_true.append(manual_data[col].values[mask])
        filtered_pred.append(pred_data[col].values[mask])

y_true_all = np.concatenate(filtered_true)
y_pred_all = np.concatenate(filtered_pred)

# Overall performance metrics
overall_accuracy  = accuracy_score(y_true_all, y_pred_all)
overall_precision = precision_score(y_true_all, y_pred_all, zero_division=0)
overall_recall    = recall_score(y_true_all, y_pred_all, zero_division=0)
overall_f1        = f1_score(y_true_all, y_pred_all, zero_division=0)
tn, fp_total, fn_total, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0,1]).ravel()

print("=== Overall Metrics ===")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")
print(f"F1 Score: {overall_f1:.4f}")
print(f"False Positives: {fp_total}")
print(f"False Negatives: {fn_total}")

# Per-class metrics
metrics_data = {
    "Class": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
    "False Positives": [],
    "False Negatives": [],
}

print("\n=== Per-Class Metrics ===")
n_rows = len(desired_idx)
for col in col_names:
    if col == "No_Biodiv":
        mask = np.ones(n_rows, dtype=bool)
    else:
        mask = ~no_biodiv_pred.values

    y_true_col = manual_data[col].values[mask]
    y_pred_col = pred_data[col].values[mask]

    acc  = accuracy_score(y_true_col, y_pred_col)
    prec = precision_score(y_true_col, y_pred_col, zero_division=0)
    rec  = recall_score(y_true_col, y_pred_col, zero_division=0)
    f1   = f1_score(y_true_col, y_pred_col, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1]).ravel()

    print(f"\nClass: {col}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")

    metrics_data["Class"].append(col)
    metrics_data["Accuracy"].append(acc)
    metrics_data["Precision"].append(prec)
    metrics_data["Recall"].append(rec)
    metrics_data["F1 Score"].append(f1)
    metrics_data["False Positives"].append(fp)
    metrics_data["False Negatives"].append(fn)

metrics_df = pd.DataFrame(metrics_data)

# === Visualization ===

# Overall Metrics with bar labels
plt.figure(figsize=(6, 5))
bars = plt.bar(["Accuracy", "Precision", "Recall", "F1 Score"],
               [overall_accuracy, overall_precision, overall_recall, overall_f1])
plt.title("Overall Classification Metrics")
plt.ylim(0, 1.05)
plt.grid(axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Per-Class Metrics with bar labels
plt.figure(figsize=(12, 16))
for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"], 1):
    plt.subplot(4, 1, i)
    bars = plt.bar(metrics_df["Class"], metrics_df[metric])
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.title(f"{metric} per Class")
    plt.ylim(0, 1.15)
    plt.grid(axis='y')
    # Add bar labels for each class
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.show()

# False Positives / Negatives
plt.figure(figsize=(14, 6))
x = np.arange(len(metrics_df["Class"]))
bar_width = 0.4
plt.bar(x - bar_width/2, metrics_df["False Positives"], width=bar_width, label='False Positives')
plt.bar(x + bar_width/2, metrics_df["False Negatives"], width=bar_width, label='False Negatives')
plt.xticks(ticks=x, labels=metrics_df["Class"], rotation=90)
plt.ylabel("Count")
plt.title("False Positives and False Negatives per Class")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Combined Chart: Overall + Per-Class Metrics (with divider)
# Safe mapping from metric name to overall value (avoid eval with spaces)
overall_map = {
    "Accuracy": overall_accuracy,
    "Precision": overall_precision,
    "Recall": overall_recall,
    "F1 Score": overall_f1,
}

plt.figure(figsize=(12, 16))
for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1 Score"], 1):
    plt.subplot(4, 1, i)

    extended_classes = ["Overall"] + list(metrics_df["Class"])
    extended_values = [overall_map[metric]] + list(metrics_df[metric])

    bars = plt.bar(extended_classes, extended_values)
    plt.xticks(rotation=90)
    plt.ylabel(metric)
    plt.title(f"{metric}: Overall and Per Class")
    plt.ylim(0, 1.15)
    plt.grid(axis='y')

    # Vertical divider between Overall (index 0) and first class (index 1)
    plt.axvline(x=0.5, color='black',  linewidth=3)

    # Add bar labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
#plt.savefig()("DeepSeek_Performance_Combined.png", dpi=300)
plt.show()


from sklearn.metrics import precision_recall_fscore_support, hamming_loss

# ----------------------------
# Helpers
# ----------------------------
def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}

def multilabel_micro_cm(Yt, Yp):
    tp = ((Yt==1)&(Yp==1)).sum()
    fp = ((Yt==0)&(Yp==1)).sum()
    fn = ((Yt==1)&(Yp==0)).sum()
    tn = ((Yt==0)&(Yp==0)).sum()
    return tn, fp, fn, tp

# ----------------------------
# 1) Relevance cohorts for gate (No_Biodiv)
#    Cohorts defined by GROUND TRUTH relevance
# ----------------------------
y_true_nb = manual_data["No_Biodiv"].values
y_pred_nb = pred_data["No_Biodiv"].values

mask_nonrel = (y_true_nb == 1)  # ground-truth non-relevant
mask_rel    = (y_true_nb == 0)  # ground-truth relevant

met_nonrel = binary_metrics(y_true_nb[mask_nonrel], y_pred_nb[mask_nonrel])
met_rel    = binary_metrics(y_true_nb[mask_rel],    y_pred_nb[mask_rel])

# Averages over the two cohorts
# Macro = unweighted mean of the two; Micro = support-weighted via concatenation
macro_gate = {
    k: (met_nonrel[k] + met_rel[k]) / 2
    for k in ["Accuracy","Precision","Recall","F1"]
}
# For micro on gate we just compute on all rows (same as your overall gate metric)
met_gate_overall = binary_metrics(y_true_nb, y_pred_nb)

# Assemble rows
rows = []
rows.append({"Section":"Gate_Cohort","Name":"NonRelevant(GT:No_Biodiv=1)", **met_nonrel,
             "Subset_Accuracy":None,"Hamming_Loss":None,"Num_Samples":int(mask_nonrel.sum()),"Num_Labels":1})
rows.append({"Section":"Gate_Cohort","Name":"Relevant(GT:No_Biodiv=0)", **met_rel,
             "Subset_Accuracy":None,"Hamming_Loss":None,"Num_Samples":int(mask_rel.sum()),"Num_Labels":1})
rows.append({"Section":"Gate_Cohort_Avg","Name":"Macro(NonRel,Rel)",
             "Accuracy":macro_gate["Accuracy"],"Precision":macro_gate["Precision"],
             "Recall":macro_gate["Recall"],"F1":macro_gate["F1"],
             "TP":None,"FP":None,"TN":None,"FN":None,
             "Subset_Accuracy":None,"Hamming_Loss":None,
             "Num_Samples":len(y_true_nb),"Num_Labels":1})
rows.append({"Section":"Gate_Cohort_Avg","Name":"Micro(NonRel,Rel)",
             **met_gate_overall,
             "Subset_Accuracy":None,"Hamming_Loss":None,
             "Num_Samples":len(y_true_nb),"Num_Labels":1})

# ----------------------------
# 2) Overall model (ALL labels incl. gate)
# ----------------------------
tn_all, fp_all, fn_all, tp_all = confusion_matrix(y_true_all, y_pred_all, labels=[0,1]).ravel()
rows.append({
    "Section":"Overall_All_Labels","Name":"All (incl. No_Biodiv)",
    "Accuracy":accuracy_score(y_true_all, y_pred_all),
    "Precision":precision_score(y_true_all, y_pred_all, zero_division=0),
    "Recall":recall_score(y_true_all, y_pred_all, zero_division=0),
    "F1":f1_score(y_true_all, y_pred_all, zero_division=0),
    "TP":tp_all,"FP":fp_all,"TN":tn_all,"FN":fn_all,
    "Subset_Accuracy":None,"Hamming_Loss":None,
    "Num_Samples":len(y_true_all),"Num_Labels":1
})

# ----------------------------
# 3) Per-class (all labels)
#     (keeps your masking rule: evaluate non-gate classes only where model predicted relevant)
# ----------------------------
n_rows = len(desired_idx)
for col in col_names:
    mask = np.ones(n_rows, dtype=bool) if col=="No_Biodiv" else ~no_biodiv_pred.values
    yt = manual_data[col].values[mask]
    yp = pred_data[col].values[mask]
    met = binary_metrics(yt, yp)
    rows.append({"Section":"Per_Class","Name":col, **met,
                 "Subset_Accuracy":None,"Hamming_Loss":None,
                 "Num_Samples":len(yt),"Num_Labels":1})

# ----------------------------
# 4) Category metrics (ACT / IMPL / ECO), excluding No_Biodiv
#     Adjust these prefix rules if your column names differ.
#     Computed on CONDITIONAL SET where model predicts relevant (gate==0).
# ----------------------------
classes_cond = [c for c in col_names if c!="No_Biodiv"]
cond_mask = ~no_biodiv_pred.values  # evaluate content only where model says relevant

ACT_CLASSES  = [c for c in classes_cond if c.upper().startswith("ACT")]
IMPL_CLASSES = [c for c in classes_cond if c.upper().startswith("IMPL")]
ECO_CLASSES  = [c for c in classes_cond if c.upper().startswith("ECO")]

category_map = {"ACT":ACT_CLASSES, "IMPL":IMPL_CLASSES, "ECO":ECO_CLASSES}

for cat, cls in category_map.items():
    if not cls:
        continue

    Yt = manual_data.loc[cond_mask, cls].values.astype(int)
    Yp = pred_data.loc[cond_mask, cls].values.astype(int)

    # Micro/macro across labels in this category
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(Yt, Yp, average="micro", zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Yt, Yp, average="macro", zero_division=0)

    tn_mi, fp_mi, fn_mi, tp_mi = multilabel_micro_cm(Yt, Yp)

    rows.append({"Section":"Category_Average","Name":f"{cat}_micro",
                 "Accuracy":None,"Precision":prec_micro,"Recall":rec_micro,"F1":f1_micro,
                 "TP":tp_mi,"FP":fp_mi,"TN":tn_mi,"FN":fn_mi,
                 "Subset_Accuracy":None,"Hamming_Loss":None,
                 "Num_Samples":Yt.shape[0],"Num_Labels":Yt.shape[1]})
    rows.append({"Section":"Category_Average","Name":f"{cat}_macro",
                 "Accuracy":None,"Precision":prec_macro,"Recall":rec_macro,"F1":f1_macro,
                 "TP":None,"FP":None,"TN":None,"FN":None,
                 "Subset_Accuracy":None,"Hamming_Loss":None,
                 "Num_Samples":Yt.shape[0],"Num_Labels":Yt.shape[1]})

    # Individual classes within the category
    for c in cls:
        yt_c = manual_data.loc[cond_mask, c].values
        yp_c = pred_data.loc[cond_mask, c].values
        met_c = binary_metrics(yt_c, yp_c)
        rows.append({"Section":"Category_Per_Class","Name":f"{cat}:{c}", **met_c,
                     "Subset_Accuracy":None,"Hamming_Loss":None,
                     "Num_Samples":len(yt_c),"Num_Labels":1})

# ----------------------------
# 5) One-sheet export
# ----------------------------
export_df = pd.DataFrame(rows, columns=[
    "Section","Name","Accuracy","Precision","Recall","F1",
    "TP","FP","TN","FN","Subset_Accuracy","Hamming_Loss","Num_Samples","Num_Labels"
])

out_path = "LLM_Performance_ONE_SHEET.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as w:
    export_df.to_excel(w, index=False, sheet_name="All_Metrics")

print(f"Saved everything to '{out_path}'")

