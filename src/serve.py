import os
 
from flask import Flask, abort
from flask import send_from_directory
 
static_file_dir = os.path.join(os.getcwd(), 'models')
app = Flask(__name__)
 
@app.route('/models/<path:path>', methods=['GET'])
def serve_file_in_dir(path):
    
    if not os.path.isfile(os.path.join(static_file_dir, path)):
        abort(404)
 
    return send_from_directory(static_file_dir, path)
 
app.run(host='0.0.0.0',port=80)
