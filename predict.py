import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

IMG_SIZE = 128


def load_image(image_path: Path, img_size: int = IMG_SIZE):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='Predict label for a single image using cnn_model.h5'
    )
    parser.add_argument(
        'image_path',
        type=Path,
        help='Path to the image file to classify'
    )
    parser.add_argument(
        '--model',
        type=Path,
        default='cnn_model.h5',
        help='Path to the saved Keras model file (default: cnn_model.h5)'
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f'Image file not found: {args.image_path}')

    if not args.model.exists():
        raise FileNotFoundError(f'Model file not found: {args.model}')

    model = tf.keras.models.load_model(str(args.model))

    image = load_image(args.image_path)
    prediction = model.predict(image, verbose=0)
    class_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction, axis=1)[0])

    class_names_path = Path('class_names.json')
    if class_names_path.exists():
        with class_names_path.open('r', encoding='utf-8') as f:
            all_classes = json.load(f)
    else:
        all_classes = sorted([
            d.name for d in Path('dataset').iterdir() if d.is_dir()
        ])

    if len(all_classes) == 0:
        print('Warning: dataset folder not found or contains no class subfolders.')
        print('Predicted class index:', class_index)
        print('Confidence:', confidence)
        return

    if class_index >= len(all_classes):
        print('Predicted class index is out of range:', class_index)
        return

    predicted_class = all_classes[class_index]

    print('Image:', args.image_path)
    print('Predicted class:', predicted_class)
    print(f'Confidence: {confidence * 100:.2f}%')


if __name__ == '__main__':
    main()
