# Şifreli Tıbbi Görüntü Sınıflandırma

Bu proje, PathMNIST veri kümesi üzerinde tıbbi görüntü sınıflandırmasını homomorfik şifreleme kullanarak gerçekleştirmeyi amaçlamaktadır.

## Proje İçeriği

- PathMNIST veri kümesi analizi
- Normal CNN modelinin eğitilmesi
- Homomorfik şifrelemeye uygun doğrusal model geliştirilmesi
- CKKS ile şifreli çıkarım
- Açık metin ve şifreli çıkarım karşılaştırması
- Performans ve sayısal hata analizi

## Veri Kümesi

- Eğitim: 89.996 görüntü
- Doğrulama: 10.004 görüntü
- Test: 7.180 görüntü
- Toplam: 107.180 görüntü
- Görüntü boyutu: 28 x 28 RGB
- Sınıf sayısı: 9

## Normal CNN Sonuçları

Normal CNN modeli test kümesinde %83,94 doğruluk ve %77,05 Macro F1 elde etmiştir.

## Homomorfik Şifreleme

Şifreli çıkarım için TenSEAL kütüphanesi ve CKKS şifreleme yöntemi kullanılmıştır.

CKKS ayarları:

- Polynomial modulus degree: 8192
- Coefficient modulus sizes: [60, 40, 40, 60]
- Global scale: 2^40

## HE Uyumlu Model

Final model 16 x 16 RGB görüntüler üzerinde çalışan doğrusal bir sınıflandırıcıdır.

Model yapısı:

16 x 16 x 3 -> Flatten -> Linear 768 -> 64 -> Linear 64 -> 9

Polynomial aktivasyonlar da denenmiş, ancak CKKS ölçek ve multiplicative-depth problemleri nedeniyle final modelde kullanılmamıştır.

## HE Model Sonuçları

Tam test kümesinde:

- Accuracy: %57,13
- Macro F1: %43,86

## Şifreli Çıkarım Benchmarkı

30 test örneği üzerinde benchmark gerçekleştirilmiştir.

- Plaintext accuracy: %56,67
- Encrypted accuracy: %50,00
- Prediction agreement: %70,00
- Ortalama maksimum CKKS hatası: 2,2001
- Ortalama mutlak hata: 0,7688
- Ortalama şifreleme süresi: 4,93 ms
- Ortalama şifreli çıkarım süresi: 921,56 ms
- Ortalama toplam HE süresi: 926,49 ms

30 örnekli benchmark sonucu modelin tamamındaki test doğruluğu olarak değerlendirilmemelidir.

## Proje Yapısı

README.md
.gitignore
requirements.txt
train_model.py
evaluate_model.py
train_he_linear.py
evaluate_he_linear_test.py
encrypted_linear_inference.py
benchmark_encrypted.py
test_ckks.py
test_encrypted_matmul.py
check_dataset_leakage.py
explore_dataset.py
plot_final_results.py

experiments/ klasörü başarısız ve deneysel HE çalışmalarını içermektedir.

## Kurulum

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Sınırlamalar

- HE modeli normal CNN modelinden daha düşük doğruluğa sahiptir.
- Şifreli benchmark yalnızca 30 örnek içermektedir.
- CKKS yaklaşık aritmetik kullanmaktadır.
- Şifreli çıkarım normal çıkarımdan çok daha yavaştır.
- Final HE modeli doğrusal işlemlerden oluşmaktadır.

## Sonuç

Proje, tıbbi görüntü sınıflandırmasında homomorfik şifreleme kullanımını deneysel olarak göstermektedir. Sonuçlar doğruluk, gizlilik, sayısal kararlılık ve hesaplama maliyeti arasında önemli bir denge bulunduğunu göstermektedir.
