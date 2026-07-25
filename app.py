import io
import torch
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

from model_def import EncoderCNNAttention, DecoderAttention

app = Flask(__name__)

# ---- Load model once at startup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('deploy_model_v2.pth', map_location=device)

word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']
MAX_LEN = checkpoint['max_len']
IMG_SIZE = checkpoint['img_size']
VOCAB_SIZE = len(word2idx)

encoder = EncoderCNNAttention().to(device)
decoder = DecoderAttention(embed_size=256, hidden_size=512, vocab_size=VOCAB_SIZE,
                            encoder_dim=2048, attention_dim=256).to(device)
encoder.load_state_dict(checkpoint['encoder_state'])
decoder.load_state_dict(checkpoint['decoder_state'])
encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_caption(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        encoder_out = encoder(image_tensor)
        h, c = decoder.init_hidden_state(encoder_out)

        word_id = word2idx['<start>']
        result_ids = []

        for _ in range(MAX_LEN):
            embed = decoder.embed(torch.tensor([word_id]).to(device))
            context, _ = decoder.attention(encoder_out, h)
            lstm_input = torch.cat([embed, context], dim=1)
            h, c = decoder.lstm_cell(lstm_input, (h, c))
            preds = decoder.fc(h)
            word_id = preds.argmax(1).item()
            result_ids.append(word_id)
            if idx2word[word_id] == '<end>':
                break

    words = [idx2word[i] for i in result_ids if idx2word[i] not in ['<start>', '<end>', '<pad>']]
    return ' '.join(words)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided, use form field "image"'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('L')
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    img_tensor = transform(img)
    caption = generate_caption(img_tensor)

    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
