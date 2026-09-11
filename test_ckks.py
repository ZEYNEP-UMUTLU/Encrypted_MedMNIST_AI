import tenseal as ts

print("CKKS testi başlıyor...")

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.global_scale = 2**40
context.generate_galois_keys()

original = [0.10, 0.20, 0.30, 0.40]

encrypted = ts.ckks_vector(
    context,
    original
)

decrypted = encrypted.decrypt()

print("Orijinal :", original)
print("Çözülmüş :", [round(x, 6) for x in decrypted])

print("\nCKKS testi başarılı.")
