import tenseal as ts
import numpy as np

print("=" * 70)
print("TENSEAL CKKS ENCRYPTED MATMUL TEST")
print("=" * 70)

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.global_scale = 2**40
context.generate_galois_keys()

# x = [1, 2, 3]
x = [1.0, 2.0, 3.0]

# W:
#
# [1 2]
# [3 4]
# [5 6]
#
W = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
]

expected = np.matmul(
    np.array(x),
    np.array(W)
)

print("\nPlaintext expected:")
print(expected)

enc_x = ts.ckks_vector(
    context,
    x
)

# encrypted x × plaintext W
enc_result = enc_x.matmul(W)

decrypted = np.array(
    enc_result.decrypt()
)

print("\nDecrypted encrypted-matmul result:")
print(decrypted)

error = np.max(
    np.abs(decrypted - expected)
)

print(f"\nMaximum absolute error: {error:.8f}")

if error < 1e-3:
    print("\nPASS: encrypted matrix multiplication çalışıyor.")
else:
    print("\nWARNING: numerical error beklenenden yüksek.")

print("=" * 70)
