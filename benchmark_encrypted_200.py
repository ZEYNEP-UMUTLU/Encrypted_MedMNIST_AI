import time
import csv
import torch
import torch.nn as nn
import numpy as np
import tenseal as ts
from torchvision import transforms
from medmnist import PathMNIST

MODEL_PATH = "he_linear_best.pth"
N_SAMPLES = 200

class_names = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium"
]

class HE_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        return self.fc2(self.fc1(x))


device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((16,16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

test_ds = PathMNIST(
    split="test",
    transform=transform,
    download=False
)

model = HE_Linear().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

W1 = model.fc1.weight.detach().cpu().numpy().T.astype(np.float64)
b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float64)

W2 = model.fc2.weight.detach().cpu().numpy().T.astype(np.float64)
b2 = model.fc2.bias.detach().cpu().numpy().astype(np.float64)


print("=" * 70)
print("ENCRYPTED MEDMNIST FINAL BENCHMARK")
print("=" * 70)
print(f"Samples: {N_SAMPLES}")
print("Scheme: CKKS")
print("TenSEAL version: 0.3.17")
print("Model: 768 -> 64 -> 9")
print()


# ---------------------------------------------------------
# CKKS CONTEXT
# ---------------------------------------------------------

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60,40,40,60]
)

context.global_scale = 2 ** 40
context.generate_galois_keys()


plaintext_correct = 0
encrypted_correct = 0
prediction_match = 0

encryption_times = []
encrypted_inference_times = []
total_encrypted_times = []

max_errors = []
mean_errors = []

rows = []


# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------

for i in range(N_SAMPLES):

    image, label = test_ds[i]

    x = image.reshape(-1).numpy().astype(np.float64)
    true_label = int(np.asarray(label).reshape(-1)[0])

    # -----------------------------
    # Plaintext inference
    # -----------------------------

    with torch.no_grad():
        x_tensor = image.reshape(1, -1).to(device)

        t0 = time.perf_counter()

        plain_logits = model(x_tensor)

        plain_time = (time.perf_counter() - t0) * 1000

        plain_logits = plain_logits.cpu().numpy()[0]

    plain_pred = int(np.argmax(plain_logits))

    # -----------------------------
    # Encryption
    # -----------------------------

    t0 = time.perf_counter()

    enc_x = ts.ckks_vector(context, x.tolist())

    encryption_time = (time.perf_counter() - t0) * 1000

    # -----------------------------
    # Encrypted inference
    # -----------------------------

    t0 = time.perf_counter()

    enc_h = enc_x.matmul(W1)
    enc_h += b1

    enc_out = enc_h.matmul(W2)
    enc_out += b2

    encrypted_time = (time.perf_counter() - t0) * 1000

    # -----------------------------
    # Decryption
    # -----------------------------

    encrypted_logits = np.array(enc_out.decrypt()[:9])

    # -----------------------------
    # Error
    # -----------------------------

    abs_error = np.abs(
        plain_logits - encrypted_logits
    )

    max_error = float(np.max(abs_error))
    mean_error = float(np.mean(abs_error))

    encrypted_pred = int(np.argmax(encrypted_logits))

    plain_ok = plain_pred == true_label
    encrypted_ok = encrypted_pred == true_label
    match = plain_pred == encrypted_pred

    if plain_ok:
        plaintext_correct += 1

    if encrypted_ok:
        encrypted_correct += 1

    if match:
        prediction_match += 1

    encryption_times.append(encryption_time)
    encrypted_inference_times.append(encrypted_time)
    total_encrypted_times.append(
        encryption_time + encrypted_time
    )

    max_errors.append(max_error)
    mean_errors.append(mean_error)

    rows.append([
        i,
        true_label,
        plain_pred,
        encrypted_pred,
        int(match),
        max_error,
        mean_error,
        encryption_time,
        encrypted_time
    ])

    print(
        f"[{i+1:02d}/{N_SAMPLES}] "
        f"true={true_label} "
        f"plain={plain_pred} "
        f"encrypted={encrypted_pred} "
        f"match={'YES' if match else 'NO'} "
        f"error={max_error:.4f}"
    )


# ---------------------------------------------------------
# FINAL STATISTICS
# ---------------------------------------------------------

plaintext_accuracy = plaintext_correct / N_SAMPLES
encrypted_accuracy = encrypted_correct / N_SAMPLES
agreement = prediction_match / N_SAMPLES

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Plaintext accuracy:        {plaintext_accuracy:.4f}")
print(f"Encrypted accuracy:        {encrypted_accuracy:.4f}")
print(f"Prediction agreement:      {agreement:.4f}")

print()
print("CKKS numerical error")
print(f"Mean maximum error:         {np.mean(max_errors):.6f}")
print(f"Mean absolute error:        {np.mean(mean_errors):.6f}")
print(f"Worst maximum error:        {np.max(max_errors):.6f}")

print()
print("TIMING")
print(f"Mean encryption time:       {np.mean(encryption_times):.4f} ms")
print(f"Mean encrypted inference:  {np.mean(encrypted_inference_times):.4f} ms")
print(f"Mean total HE time:         {np.mean(total_encrypted_times):.4f} ms")

print()
print(f"Plaintext correct:          {plaintext_correct}/{N_SAMPLES}")
print(f"Encrypted correct:          {encrypted_correct}/{N_SAMPLES}")
print(f"Prediction matches:        {prediction_match}/{N_SAMPLES}")

print()
print("=" * 70)

if prediction_match == N_SAMPLES:
    print("PREDICTION AGREEMENT: PASS")
else:
    print("PREDICTION AGREEMENT: PARTIAL")

if encrypted_correct == plaintext_correct:
    print("ACCURACY CONSISTENCY: PASS")
else:
    print("ACCURACY CONSISTENCY: CHECK")

print("=" * 70)


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

with open("encrypted_benchmark_200.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample",
        "true_label",
        "plaintext_prediction",
        "encrypted_prediction",
        "prediction_match",
        "max_abs_error",
        "mean_abs_error",
        "encryption_ms",
        "encrypted_inference_ms"
    ])

    writer.writerows(rows)

print()
print("Saved: encrypted_benchmark_200.csv")
