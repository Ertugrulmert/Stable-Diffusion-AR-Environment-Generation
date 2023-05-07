import os, sys
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename

server_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from server.arcore_handler import ARCoreHandler

UPLOAD_FOLDER = os.path.join(server_dir, "user_data")
RELATIVE_UPLOAD_FOLDER = "user_data"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

handler = ARCoreHandler(data_root=UPLOAD_FOLDER)


# to display the connection status
@app.route('/', methods=['GET'])
def handle_call():
    print("Successfully Connected")
    return "Successfully Connected"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request)
        print(request.files)

    timestamp = 0

    if request.form.get("timestamp") is not None:
        timestamp = int(request.form.get("timestamp"))

    if request.files.get("rgbImage") is not None and request.files.get("depthImage") is not None:
        f = request.files["rgbImage"]
        rgb_filename = secure_filename(f.filename)
        rgb_filepath = os.path.join(app.config['UPLOAD_FOLDER'], rgb_filename)
        f.save(rgb_filepath)

        f = request.files["depthImage"]
        depth_filename = secure_filename(f.filename)
        depth_filepath = os.path.join(app.config['UPLOAD_FOLDER'], depth_filename)
        f.save(depth_filepath)

        mesh_name, material_name, texture_name = handler.process_arcore_ground(rgb_filepath, depth_filepath, i=timestamp)

        return jsonify(isError=False,
                       message="Success, Mesh Created",
                       timestamp=timestamp,
                       mesh_path=mesh_name,
                       material_path=material_name,
                       texture_path=texture_name,
                       statusCode=200), 200

    else:
        return jsonify(
            message="Connection successful.",
            category="success",
            status=200)


@app.route('/mesh/<path:mesh_path>', methods=['GET'])
def get_mesh_file(mesh_path):
    mesh_buf = handler.get_serialized_object(mesh_path)
    #mesh_buf = handler.get_serialized_object("andy.obj")

    if mesh_buf is not None:

        # return Response(mesh_buf, mimetype="text/plain"), 200
        return Response(mesh_buf, mimetype="application/wavefront-obj"), 200
        #return Response(mesh_buf, mimetype="model/obj"), 200


    else:
        return Response(
            "Mesh File not found",
            status=400,
        )


@app.route('/mat/<path:material_path>', methods=['GET'])
def get_material_file(material_path):
    material_buf = handler.get_serialized_object(material_path)

    if material_buf is not None:

        # return Response(mesh_buf, mimetype="text/plain"), 200
        # return Response(mesh_buf, mimetype="application/wavefront-obj"), 200
        return Response(material_buf, mimetype="model/mtl"), 200


    else:
        return Response(
            "Material File not found",
            status=400,
        )

@app.route('/texture/<path:texture_path>', methods=['GET'])
def get_texture_file(texture_path):
    texture_buf = handler.get_serialized_object(texture_path)

    if texture_buf is not None:
        return Response(texture_buf, mimetype="image/png"), 200

    else:
        return Response(
            "Texture File not found",
            status=400,
        )


@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host="0.0.0.0", port=5000, debug=True)
