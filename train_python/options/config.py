import yaml


class Config:
    def __init__(self, path):

        with open(path, 'r') as f:
            params = yaml.safe_load(f)

            self.input_channel = params["model"]["Input_channel"]  # type int
            self.image_size = params["model"]["Image_size"]  # type list[w, h]
            self.kernel_size = params["model"]["K_size"]  # type int
            self.classes = params["model"]["class"]  # type int

            self.train_csv = params["train"]["train_csv"]  # type list
            self.test_csv = params["train"]["test_csv"]  # type list

            self.epochs = params["train"]["Epoch"]  # type int
            self.batch_size = params["train"]["BatchSize"]  # type int

            # type string
            self.saved_path = params["train"]["saved_path"]
            self.traced_path = params["train"]["traced_path"]

