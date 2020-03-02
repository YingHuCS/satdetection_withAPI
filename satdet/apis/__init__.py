# Added on 2020-02-28
from flask_restplus import Api
from .server_path import ns as ns1
from core import argsparser
import os, logging.config

args = argsparser.prepare_args()

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
if os.environ['LOGLEVEL'] == 'DEBUG':
    logging.info('Set log level to DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    logging.info('Set log level to INFO')
    logger.setLevel(logging.INFO)

api = Api(
    title='Ying Hu Project 1 RestAPI',
    version='1.0.1',
    description='API to xxxxx xxxx.',
)

if args.activate_detection:
    api.add_namespace(ns1)
