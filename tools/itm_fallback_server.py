from flask import Flask, jsonify, request

app = Flask(__name__)

@app.post('/blip2itm')
def blip2itm():
    # Temporary deterministic ITM fallback for closing the ROS/Habitat/Jev loop.
    # It does not create candidates or choose actions; Jev remains the high-level selector.
    if not isinstance(request.json, dict) or 'image' not in request.json or 'txt' not in request.json:
        return jsonify({'error': 'image and txt are required'}), 400
    return jsonify({'response': 0.5, 'itm score': 0.5, 'fallback': True})

if __name__ == '__main__':
    app.run(host='localhost', port=12182)
