import tenseal as ts

print("=" * 70)
print("TENSEAL API CHECK")
print("=" * 70)

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.global_scale = 2**40
context.generate_galois_keys()

enc = ts.ckks_vector(
    context,
    [1.0, 2.0, 3.0, 4.0]
)

print("\nTenSEAL version:")
print(ts.__version__)

print("\nCKKSVector methods relevant to matrix/vector operations:")

methods = [
    x for x in dir(enc)
    if any(
        key in x.lower()
        for key in [
            "matmul",
            "dot",
            "mm",
            "mul",
            "add"
        ]
    )
]

for method in methods:
    print(" ", method)

print("\nCKKSVector type:")
print(type(enc))

print("\nAPI check tamamlandı.")
