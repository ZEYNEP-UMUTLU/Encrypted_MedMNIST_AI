import time
import torch
import torch.nn as nn
import numpy as np
import tenseal as ts
from torchvision import transforms
from medmnist import PathMNIST

MODEL_PATH = "he_linear_best.pth"

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

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.fc2(x)

        return x


model = HE_Linear()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((16,16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])


dataset = PathMNIST(
    split="val",
    transform=transform,
    download=False
)


# --------------------------------------------------
# Select one test image
# --------------------------------------------------

image, label = dataset[0]

label = int(label[0])

x = image.view(-1).numpy().astype(np.float64)


# --------------------------------------------------
# Plaintext inference
# --------------------------------------------------

x_tensor = torch.tensor(
    x,
    dtype=torch.float32
).unsqueeze(0)


start = time.perf_counter()

with torch.no_grad():

    plaintext_logits = model(
        x_tensor
    ).squeeze(0).numpy()

plaintext_time = (
    time.perf_counter() - start
) * 1000


plaintext_prediction = int(
    np.argmax(plaintext_logits)
)


# --------------------------------------------------
# Extract weights
# --------------------------------------------------

W1 = (
    model.fc1.weight
    .detach()
    .numpy()
    .T
    .astype(np.float64)
)

b1 = (
    model.fc1.bias
    .detach()
    .numpy()
    .astype(np.float64)
)

W2 = (
    model.fc2.weight
    .detach()
    .numpy()
    .T
    .astype(np.float64)
)

b2 = (
    model.fc2.bias
    .detach()
    .numpy()
    .astype(np.float64)
)


# --------------------------------------------------
# CKKS context
#
# This is the configuration already verified
# to produce ~1e-6 FC1 error.
# --------------------------------------------------

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60,40,40,60]
)

context.global_scale = 2**40

context.generate_galois_keys()


# --------------------------------------------------
# Encrypt input
# --------------------------------------------------

start = time.perf_counter()

enc_x = ts.ckks_vector(
    context,
    x.tolist()
)

encryption_time = (
    time.perf_counter() - start
) * 1000


# --------------------------------------------------
# Encrypted FC1
# --------------------------------------------------

enc_h = enc_x.matmul(W1)

enc_h = enc_h + b1.tolist()


# --------------------------------------------------
# Encrypted FC2
# --------------------------------------------------

start = time.perf_counter()

enc_out = enc_h.matmul(W2)

enc_out = enc_out + b2.tolist()

encrypted_logits = np.array(
    enc_out.decrypt()
)

encrypted_time = (
    time.perf_counter() - start
) * 1000


# --------------------------------------------------
# Prediction
# --------------------------------------------------

encrypted_prediction = int(
    np.argmax(encrypted_logits)
)


# --------------------------------------------------
# Error analysis
# --------------------------------------------------

max_error = np.max(
    np.abs(
        encrypted_logits
        - plaintext_logits
    )
)

mean_error = np.mean(
    np.abs(
        encrypted_logits
        - plaintext_logits
    )
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 70)
print("ENCRYPTED MEDMNIST INFERENCE")
print("=" * 70)

print()

print("True label:")
print(
    f"{label} - {class_names[label]}"
)

print()

print("Plaintext prediction:")
print(
    f"{plaintext_prediction} - "
    f"{class_names[plaintext_prediction]}"
)

print()

print("Encrypted prediction:")
print(
    f"{encrypted_prediction} - "
    f"{class_names[encrypted_prediction]}"
)

print()

print("Plaintext logits:")
print(plaintext_logits)

print()

print("Encrypted logits:")
print(encrypted_logits)

print()

print("=" * 70)
print("CKKS NUMERICAL ERROR")
print("=" * 70)

print(
    f"Maximum absolute error: "
    f"{max_error:.8f}"
)

print(
    f"Mean absolute error: "
    f"{mean_error:.8f}"
)

print()

print("=" * 70)
print("TIMING")
print("=" * 70)

print(
    f"Plaintext inference: "
    f"{plaintext_time:.4f} ms"
)

print(
    f"Encryption: "
    f"{encryption_time:.4f} ms"
)

print(
    f"Encrypted inference: "
    f"{encrypted_time:.4f} ms"
)

print()

print("=" * 70)

if plaintext_prediction == encrypted_prediction:

    print("PREDICTION MATCH: PASS")

else:

    print("PREDICTION MATCH: FAIL")

print("=" * 70)
