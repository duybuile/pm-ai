"""Saturn PM assistant package."""
from dotenv import find_dotenv, load_dotenv

from src.conf import Config

load_dotenv(find_dotenv())
cfg = Config('conf')
