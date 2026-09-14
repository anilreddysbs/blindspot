"""Generate demo datasets with deliberately hidden patterns for BlindSpot."""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# ---------------- students ----------------
n = 180
ids = [f"S{i:03d}" for i in range(1, n + 1)]
section = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.3, 0.3])

attendance, assign, internal, lab, final, hours = [], [], [], [], [], []
for i in range(n):
    sec = section[i]
    # anomaly group: first 24 students -> high everything, low final, low study hours
    if i < 24:
        a = rng.normal(92, 3); asg = rng.normal(95, 3); im = rng.normal(88, 3)
        lb = rng.normal(90, 3); fh = rng.normal(50, 3); sh = rng.normal(1.7, 0.4)
    else:
        base = rng.normal(75, 10)
        a = base + rng.normal(8, 4); asg = base + rng.normal(10, 5)
        im = base + rng.normal(5, 5); lb = base + rng.normal(6, 5)
        sh = rng.normal(4.2, 1.0)
        fh = 0.45 * im + 0.25 * asg + 2.5 * sh + rng.normal(0, 4)
        if sec == "B":  # Section B anomaly: -14% final despite similar inputs
            fh -= 12
    attendance.append(round(float(np.clip(a, 35, 100)), 1))
    assign.append(round(float(np.clip(asg, 30, 100)), 1))
    internal.append(round(float(np.clip(im, 25, 100)), 1))
    lab.append(round(float(np.clip(lb, 25, 100)), 1))
    final.append(round(float(np.clip(fh, 20, 100)), 1))
    hours.append(round(float(max(0.3, sh)), 1))

students = pd.DataFrame({
    "Student_ID": ids, "Attendance": attendance, "Assignments": assign,
    "Internal_Marks": internal, "Lab_Marks": lab, "Final_Marks": final,
    "Study_Hours": hours, "Section": section,
})
# missing attendance for ~9% + one data-entry error
miss = rng.choice(n, size=16, replace=False)
students.loc[miss, "Attendance"] = np.nan
students.loc[5, "Attendance"] = 147  # deliberate error
students.to_csv(OUT / "students.csv", index=False)

# ---------------- sales ----------------
months = ["January", "February", "March", "April", "May", "June"]
products = ["Alpha", "Beta", "Gamma"]
regions = ["North", "South", "East", "West"]
rows = []
rid = 1
for m_i, m in enumerate(months):
    for _ in range(30):
        p = rng.choice(products); r = rng.choice(regions)
        traffic = rng.normal(1000, 120)
        spend = rng.normal(50000, 5000) + (8000 if m == "March" else 0)
        discount = rng.normal(10, 3) + (5 if m == "March" else 0)
        delay = rng.normal(2.5, 0.6) + (1.6 if m == "March" else 0)
        sales = 90000 + traffic * 40 - delay * 9000 - discount * 200 + rng.normal(0, 6000)
        if m == "March":
            sales -= 22000  # hidden drop despite stable traffic + higher spend
        rows.append([f"T{rid:04d}", m, p, r, round(max(20000, sales)),
                     int(np.clip(rng.normal(40, 12) + (25 if m == "March" else 0), 0, 200)),
                     round(max(0, discount), 1), round(max(0.5, delay), 1),
                     round(max(1, min(5, rng.normal(4.1, 0.5) - (0.7 if m == "March" else 0))), 1),
                     int(traffic), int(spend)])
        rid += 1
sales_df = pd.DataFrame(rows, columns=["Txn_ID", "Month", "Product", "Region", "Sales",
                                       "Returns", "Discount", "Delivery_Time",
                                       "Customer_Rating", "Traffic", "Marketing_Spend"])
sales_df.to_csv(OUT / "sales.csv", index=False)
print(f"wrote {OUT/'students.csv'} ({len(students)} rows), {OUT/'sales.csv'} ({len(sales_df)} rows)")
