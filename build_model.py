from model_loader import ModelLoader

if __name__ == '__main__':
    loader = ModelLoader()
    model = loader.build_model()
    print("Model berhasil dibangun dan disimpan.")