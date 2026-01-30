
---

# EMNIST Diffüzyon Modeli: Eğitim ve Üretim Rehberi

Bu depo, PyTorch ve Hugging Face `diffusers` kütüphanesini kullanarak EMNIST veri kümesi üzerinde bir **Görüntü Üretici Diffüzyon Modeli** eğitmek ve bu modelle yeni görüntüler oluşturmak için gerekli kodları içerir.

## 1. Sistemin Çalışma Mantığı

### Eğitim Süreci (`eğitim.ipynb`)

* **Veri Kümesi:** NIST el yazısı karakterlerinden türetilen EMNIST veri kümesi kullanılır. Görüntüler 28x28 boyutundadır.
* **Model:** Bir `UNet2DModel` mimarisi kullanılır. Bu model, gürültülü bir görüntüden gürültüyü tahmin etmeyi öğrenir.
* **Gürültü Zaman Çizelgesi (Scheduler):** `DDPMScheduler` kullanılarak görüntülere aşamalı olarak gürültü eklenir (Forward Diffusion).
* **Hızlandırma:** `accelerate` kütüphanesi sayesinde eğitim GPU üzerinde optimize edilir.
* **Kayıt:** Her epoch sonunda modelin durumu (`state`) ve performans metrikleri (SSIM, FID, IS) kaydedilir.

### Üretim Süreci (`üretim.ipynb`)

* **Ters Diffüzyon:** Model, saf gürültüden (Gaussian Noise) başlayarak eğitilen `UNet2DModel` yardımıyla gürültüyü adım adım temizler ve anlamlı bir karakter görüntüsü oluşturur.
* **Model Yükleme:** Belirli bir epoch'ta (örneğin Epoch 50) kaydedilmiş olan model ağırlıkları yüklenerek üretim yapılır.

---

## 2. Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanın:

```bash
pip install torch torchvision diffusers accelerate pillow pandas torchmetrics

```

---

## 3. Epoch 50 Kullanarak Üretim Yapmak

Eğer eğitiminizi tamamladıysanız ve 50. epoch'taki modeli kullanarak üretim yapmak istiyorsanız, `üretim.ipynb` dosyasındaki mantığı şu şekilde kullanabilirsiniz:

### Adım 1: Modeli Yükleme

Eğitim sırasında modeller `emnist_diffusion_model/epoch_50` klasörüne kaydedilir. Bu ağırlıkları şu şekilde yükleriz:

```python
from diffusers import UNet2DModel, DDPMScheduler
import torch
from accelerate import Accelerator

accelerator = Accelerator()
output_dir = "emnist_diffusion_model"
current_epoch = 50  # Kullanmak istediğiniz epoch

# Modeli ve Scheduler'ı tanımla
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
)

# 50. Epoch ağırlıklarını yükle
model_path = f"{output_dir}/epoch_{current_epoch}"
accelerator.load_state(model_path)
model = accelerator.prepare(model)
model.eval() # Modeli değerlendirme moduna al

```

### Adım 2: Görüntü Üretme

Model yüklendikten sonra, rastgele gürültüden karakter üretmek için `DDPMScheduler` kullanılır:

```python
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
sample = torch.randn((8, 1, 28, 28)).to(accelerator.device) # 8 adet rastgele gürültü

for t in noise_scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(sample, t).sample
    sample = noise_scheduler.step(noisy_residual, t, sample).prev_sample

# 'sample' artık üretilen 8 adet EMNIST benzeri görüntüyü içerir.

```

---

## 4. Dosya Yapısı

* `emnist_diffusion_model/`: Eğitim sırasında kaydedilen tüm model checkpoint'leri ve görseller burada saklanır.
* `evaluation_metrics_epoch_50_random_seed.csv`: 50. epoch için hesaplanan SSIM, FID ve IS skorlarını içerir.

## 5. Önemli Notlar

* **Bellek Kullanımı:** Eğitim sırasında `Accelerator` kullanımı bellek yönetimini kolaylaştırır ancak yüksek `batch_size` değerleri GPU belleğini zorlayabilir.
* **Değerlendirme:** Üretilen görüntülerin kalitesi `üretim.ipynb` içindeki `show_label_aligned_images` fonksiyonu ile görselleştirilebilir.







Harika bir fikir. Kullanıcıların modelin başarısını görsel olarak görebilmeleri için README dosyasına bir **"Örnek Çıktılar ve Performans"** bölümü eklemek projenin etkileyiciliğini artıracaktır.

İşte README dosyana ekleyebileceğin ilgili bölüm:

---

## 6. Örnek Çıktı
 
<img width="5200" height="200" alt="label_aligned_samples_epoch_50_run_1" src="https://github.com/user-attachments/assets/84ccec08-6f06-4cff-94f5-ea1b7391048e" />









