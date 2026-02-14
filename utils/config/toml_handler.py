import toml


# Load the TOML configuration file
def load_toml(file_path):
    with open(file_path, 'r') as f:
        config = toml.load(f)
    return config
