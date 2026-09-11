import time
import torch
import torch.nn as nn
from torchvision import transforms
from medmnist import PathMNIST
import tenseal as ts
import numpy as np

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "he_mlp_best.pth"

device = torch.device("cpu")

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

# ============================================================
# MODEL
# ============================================================

class PolynomialActivation(nn.Module):

    def forward(self, x):
        return 0.5 * x + 0.1 * x * x


class HE_MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 64)
        self.poly = PolynomialActivation()
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        x = self.poly(x)

        x = self.fc2(x)

        return x


model = HE_MLP().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

# ============================================================
# DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = PathMNIST(
    split="val",
    transform=transform,
    download=False
)

image, label = dataset[0]

label = int(label[0])

x = image.view(-1).numpy().astype(np.float64)

# ============================================================
# PLAINTEXT INFERENCE
# ============================================================

x_tensor = torch.tensor(
    x,
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():

    plaintext_logits = model(
        x_tensor
    )

plaintext_logits = (
    plaintext_logits
    .squeeze()
    .numpy()
)

plaintext_prediction = int(
    np.argmax(plaintext_logits)
)

# ============================================================
# CKKS CONTEXT
# ============================================================

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60,40,40,40,40,60]
)

context.global_scale = 2**30

context.generate_galois_keys()

# ============================================================
# EXTRACT MODEL PARAMETERS
# ============================================================

W1 = (
    model.fc1.weight
    .detach()
    .cpu()
    .numpy()
    .T
    .astype(np.float64)
)

b1 = (
    model.fc1.bias
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

W2 = (
    model.fc2.weight
    .detach()
    .cpu()
    .numpy()
    .T
    .astype(np.float64)
)

b2 = (
    model.fc2.bias
    .detach()
    .cpu()
    .numpy()
    .astype(np.float64)
)

# ============================================================
# ENCRYPTED INFERENCE
# ============================================================

start = time.perf_counter()

enc_x = ts.ckks_vector(
    context,
    x.tolist()
)

# encrypted FC1 using column-wise dot products
# TenSEAL matmul() is avoided because its matrix packing
# produced incorrect FC1 values for this model.
enc_h = [
    enc_x.dot(W1[:, j].tolist())
    for j in range(W1.shape[1])
]

enc_h = [
    enc_h[j] + float(b1[j])
    for j in range(len(enc_h))
]

# FC1 bias is already included above.

print("\n[DEBUG] After FC1 + bias")

enc_h_plain = np.array([
    float(v.decrypt()[0])
    for v in enc_h
])

print("TenSEAL FC1 first 5:", enc_h_plain[:5])

with torch.no_grad():
    plain_h = model.fc1(
        torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    ).squeeze(0).numpy()

print("PyTorch FC1 first 5:", plain_h[:5])
print("FC1 max error:", np.max(np.abs(enc_h_plain - plain_h)))

# FC1 diagnostic only
# FC2 is temporarily skipped until FC1 correctness is confirmed.

print("\n[DEBUG] FC1 diagnostic complete.")
print("Expected FC1 max error should be very small.")

raise SystemExit

# add final bias
enc_out = enc_out + b2.tolist()

# add final bias
enc_out = enc_out + b2.tolist()

# decrypt
encrypted_logits = np.array(
    enc_out.decrypt()
)

encrypted_time = (
    time.perf_counter() - start
)

encrypted_prediction = int(
    np.argmax(encrypted_logits)
)

# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("PATHMNIST CKKS ENCRYPTED INFERENCE")
print("=" * 70)

print("\nTrue class:")
print(
    f"{label} - {class_names[label]}"
)

print("\nPlaintext prediction:")
print(
    f"{plaintext_prediction} - "
    f"{class_names[plaintext_prediction]}"
)

print("\nEncrypted prediction:")
print(
    f"{encrypted_prediction} - "
    f"{class_names[encrypted_prediction]}"
)

print("\nPlaintext logits:")
print(
    np.round(
        plaintext_logits,
        4
    )
)

print("\nDecrypted encrypted logits:")
print(
    np.round(
        encrypted_logits,
        4
    )
)

max_error = np.max(
    np.abs(
        plaintext_logits
        - encrypted_logits
    )
)

print("\nMaximum logit error:")
print(
    f"{max_error:.8f}"
)

print("\nEncrypted inference time:")
print(
    f"{encrypted_time * 1000:.2f} ms"
)

print("\nPrediction agreement:")

if plaintext_prediction == encrypted_prediction:
    print("PASS - Plaintext == Encrypted")
else:
    print("WARNING - Predictions differ")

print("\n" + "=" * 70)
