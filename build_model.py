from model_loader import ModelLoader # Mengimpor class ModelLoader dari file model_loader.py

# Mengeksekusi kode hanya jika file ini dijalankan secara langsung
if __name__ == '__main__':
    # Membuat objek dari class ModelLoader
    loader = ModelLoader()

    # Membangun model dari data dan menyimpannya ke file
    model = loader.build_model()

    # Menampilkan pesan bahwa model berhasil dibuat dan disimpan
    print("Model berhasil dibangun dan disimpan.")