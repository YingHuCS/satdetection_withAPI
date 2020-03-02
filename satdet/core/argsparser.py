# Added on 2020-02-20
import argparse
from .envdefault import EnvDefault

def prepare_args():
    parser = argparse.ArgumentParser(description="Ying Hu Project 1")
    parser.add_argument('--activate_detection', action=EnvDefault, envvar='ACTIVATE_DETECTION', type=bool, default=True)
    parser.add_argument('--host', action=EnvDefault, envvar='HOST', type=str, default='0.0.0.0')
    parser.add_argument('--port', action=EnvDefault, envvar='PORT', type=int, default=5701)
    args = parser.parse_args()
    return args
