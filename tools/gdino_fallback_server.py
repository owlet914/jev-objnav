from flask import Flask, jsonify, request

app = Flask(__name__)

@app.post('/gdino')
def gdino():
    if not isinstance(request.json, dict):
        return jsonify({'error': 'json body required'}), 400
    return jsonify({'boxes': [], 'logits': [], 'phrases': [], 'fallback': True})

if __name__ == '__main__':
    app.run(host='localhost', port=12181)
