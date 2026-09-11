import torch
import numpy as np
import tenseal as ts
from torchvision import transforms
from medmnist import PathMNIST

MODEL_PATH = "he_mlp_best.pth"

class HE_MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(768, 64)
        self.fc2 = torch.nn.Linear(64, 9)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = 0.5*x + 0.1*x*x
        x = self.fc2(x)
        return x

model = HE_MLP()

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

image, label = dataset[0]

x = image.view(-1).numpy().astype(np.float64)

W1 = model.fc1.weight.detach().numpy().T.astype(np.float64)
b1 = model.fc1.bias.detach().numpy().astype(np.float64)

# -----------------------------
# Plaintext reference
# -----------------------------

with torch.no_grad():

    plain_h = model.fc1(
        torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    ).squeeze(0).numpy()

    plain_poly = (
        0.5 * plain_h
        + 0.1 * plain_h * plain_h
    )

print("=" * 70)
print("PLAINTEXT REFERENCE")
print("=" * 70)

print("FC1 first 5:")
print(plain_h[:5])

print("Polynomial first 5:")
print(plain_poly[:5])

# -----------------------------
# CKKS
# -----------------------------

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60,30,30,30,60]
)

context.global_scale = 2**20
context.generate_galois_keys()

enc_x = ts.ckks_vector(
    context,
    x.tolist()
)

# -----------------------------
# Encrypted FC1
# -----------------------------

enc_h = enc_x.matmul(W1)
enc_h = enc_h + b1.tolist()

dec_h = np.array(enc_h.decrypt())

print()
print("=" * 70)
print("ENCRYPTED FC1")
print("=" * 70)

print("TenSEAL FC1 first 5:")
print(dec_h[:5])

print("FC1 max error:")
print(np.max(np.abs(dec_h - plain_h)))

print("FC1 mean absolute error:")
print(np.mean(np.abs(dec_h - plain_h)))

# -----------------------------
# Polynomial activation
# -----------------------------

print()
print("=" * 70)
print("TESTING POLYNOMIAL ACTIVATION")
print("=" * 70)

try:

    enc_h_poly = (
        enc_h * 0.5
        + enc_h.square() * 0.1
    )

    dec_poly = np.array(
        enc_h_poly.decrypt()
    )

    print("TenSEAL polynomial first 5:")
    print(dec_poly[:5])

    print("PyTorch polynomial first 5:")
    print(plain_poly[:5])

    print()
    print("Polynomial max error:")
    print(
        np.max(
            np.abs(
                dec_poly - plain_poly
            )
        )
    )

    print("Polynomial mean absolute error:")
    print(
        np.mean(
            np.abs(
                dec_poly - plain_poly
            )
        )
    )

    print()
    print("POLYNOMIAL TEST: PASS")

except Exception as e:

    print()
    print("POLYNOMIAL TEST: FAILED")
    print(type(e).__name__ + ":", e)

