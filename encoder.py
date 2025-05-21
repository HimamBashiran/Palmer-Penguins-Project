import pandas as pd

# Kelas untuk melakukan encoding data input pengguna agar sesuai dengan format model
class Encoder:
    def __init__(self, input_df, dataset_path):
        self.input_df = input_df  # Data input dari pengguna
        self.dataset_path = dataset_path  # Path ke dataset utama
        self.encoded_df = None  # Menyimpan hasil encoding

    def encode_input(self):
        # Membaca dataset
        penguins_raw = pd.read_csv(self.dataset_path) 
        penguins = penguins_raw.drop(columns=['species']) # Menghapus kolom target 'species' karena tidak dibutuhkan untuk encoding
        
        # Menggabungkan input pengguna dengan dataset untuk memastikan encoding konsisten
        df = pd.concat([self.input_df, penguins], axis=0)

        # Melakukan one-hot encoding pada kolom kategorikal
        encode = ['sex', 'island']
        for col in encode:
            # Membuat kolom dummy untuk setiap kategori
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy], axis=1) # Gabungkan dengan dataframe utama
            del df[col] # Hapus kolom asli karena sudah di-encode

        # Mengambil kembali hanya baris input pengguna setelah encoding
        self.encoded_df = df[:1]
        return self.encoded_df # Kembalikan DataFrame yang sudah diencoding