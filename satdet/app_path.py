# Added on 2020-02-28
from flask import Flask
from apis import api
from core import argsparser
args = argsparser.prepare_args()

app = Flask(__name__)

app.config.update(PROPAGATE_EXCEPTIONS=True)
api.init_app(app)
app.run(args.host, args.port, use_reloader=False)
