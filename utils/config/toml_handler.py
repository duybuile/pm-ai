import toml


# Load the TOML configuration file
def load_toml(file_path):
    # TOML config files in this project are UTF-8 encoded.
    # On Windows, relying on default encoding (e.g. cp1252) can fail on emoji/unicode.
    with open(file_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    return config
