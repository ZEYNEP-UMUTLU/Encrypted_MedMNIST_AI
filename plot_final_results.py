import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("encrypted_benchmark_30.csv")

# ---------------------------------------------------------
# 1. CKKS numerical error
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["sample"], df["max_abs_error"], marker="o")
plt.axhline(
    df["max_abs_error"].mean(),
    linestyle="--",
    label=f"Mean = {df['max_abs_error'].mean():.4f}"
)
plt.xlabel("Sample")
plt.ylabel("Maximum absolute logit error")
plt.title("CKKS Numerical Error Across 30 Test Samples")
plt.legend()
plt.tight_layout()
plt.savefig("ckks_error_30.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 2. Timing
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(
    df["sample"],
    df["encryption_ms"],
    marker="o",
    label="Encryption"
)
plt.plot(
    df["sample"],
    df["encrypted_inference_ms"],
    marker="o",
    label="Encrypted inference"
)
plt.xlabel("Sample")
plt.ylabel("Time (ms)")
plt.title("Encrypted Inference Timing Across 30 Test Samples")
plt.legend()
plt.tight_layout()
plt.savefig("he_timing_30.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 3. Prediction agreement
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
counts = df["prediction_match"].value_counts().sort_index()

plt.bar(
    ["Mismatch", "Match"],
    [counts.get(0, 0), counts.get(1, 0)]
)
plt.ylabel("Number of samples")
plt.title("Plaintext vs Encrypted Prediction Agreement")
plt.tight_layout()
plt.savefig("prediction_agreement_30.png", dpi=200)
plt.close()

print("=" * 60)
print("FINAL PLOTS CREATED")
print("=" * 60)
print("CKKS mean max error:", df["max_abs_error"].mean())
print("CKKS mean absolute error:", df["mean_abs_error"].mean())
print("Mean encryption:", df["encryption_ms"].mean(), "ms")
print("Mean encrypted inference:", df["encrypted_inference_ms"].mean(), "ms")
print("Prediction agreement:", df["prediction_match"].mean())
print()
print("Saved:")
print("  ckks_error_30.png")
print("  he_timing_30.png")
print("  prediction_agreement_30.png")
