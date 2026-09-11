import pandas as pd
import matplotlib.pyplot as plt

rand = pd.read_csv("encrypted_benchmark_200.csv")
bal = pd.read_csv("encrypted_benchmark_balanced_180.csv")

labels = ["200 Random", "180 Balanced"]

plain = [
    (rand.plaintext_prediction == rand.true_label).mean()*100,
    (bal.plaintext_prediction == bal.true_label).mean()*100
]

enc = [
    (rand.encrypted_prediction == rand.true_label).mean()*100,
    (bal.encrypted_prediction == bal.true_label).mean()*100
]

agree = [
    rand.prediction_match.mean()*100,
    bal.prediction_match.mean()*100
]

plt.figure(figsize=(8,5))
x = range(len(labels))
w = 0.25

plt.bar([i-w for i in x], plain, width=w, label="Plaintext")
plt.bar(x, enc, width=w, label="Encrypted")
plt.bar([i+w for i in x], agree, width=w, label="Agreement")

plt.xticks(list(x), labels)
plt.ylabel("Percentage (%)")
plt.ylim(0,100)
plt.title("CKKS Benchmark Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=200)

print("\n=== SUMMARY ===")
for i,l in enumerate(labels):
    print(f"{l}: Plain={plain[i]:.2f}%  Enc={enc[i]:.2f}%  Agr={agree[i]:.2f}%")

print("\nSaved: benchmark_comparison.png")
