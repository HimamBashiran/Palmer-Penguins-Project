import pickle

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
        return self.model
