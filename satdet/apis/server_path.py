# Added on 2020-02-28
from core import argsparser
from flask_restplus import Namespace, Resource, fields
from inference_funcs import *
import logging
import types

ns = Namespace('', description='Ying Hu Project 1 Descriptions')

args = argsparser.prepare_args()

parser = ns.parser()
parser.add_argument('path', type=str, location='json', default='', help='Input single file path.')

model = ns.model('Model', {'message': fields.String, 'final_result': fields.List(fields.List(fields.List(fields.Float)))})

@ns.route('/inference_single')
@ns.expect(parser)
class InferenceSingle(Resource):
    def __init__(self, *args, **kwargs):
        super(InferenceSingle, self).__init__(*args, **kwargs)

    @ns.marshal_with(model)
    def post(self):
        arguments = parser.parse_args()

        if arguments.path is None:
            res = {'message': 'Missing path attribute'}, 500
            return res

        path_str = arguments["path"]
        logging.debug("path: " + path_str)

        try:
            result = inference_single_func(path_str)
            if result is not None: #and type(result) is types.DictType:
                res = {'message':'OK', 'final_result': result}, 200
            else:
                res = {'message':'OK', 'final_result': None}, 200
            return res
        except IOError as e:
            res = {'message': str(e)}, 500
            return res
        except Exception as e:
            res = {'message': 'Internal Errorr: ' + str(e)}, 500
            return res
