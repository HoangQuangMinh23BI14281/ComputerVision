import os
from PIL import Image
import json

def parse_annotation(annotation):
    boxes = list()
    texts = list()

    with open(annotation, 'r') as f:
        for line in f.readlines():
            line_spl = line.split(',')

            full_box = [int(line_spl[i]) for i in range(8)]
            box_len = ' '.join([str(element) for element in full_box])
            text = line[len(box_len) + 1:-1]

            box = [full_box[0], full_box[1], full_box[4], full_box[5]]

            boxes.append(box)
            texts.append(text)

    return boxes, texts


def create_data(data_directory='data/raw_data/train',
                image_directory='data/task2/image', annotation_directory='data/task2/annotation'):
    all_image_paths = list()
    all_texts = list()

    index = 0
    for file in os.listdir(data_directory):
        if file.endswith('.txt'):
            if file[: -4] + '.jpg' not in os.listdir(data_directory):
                continue

            annotation = os.path.join(os.getcwd(), data_directory, file)
            image_path = os.path.join(os.getcwd(), data_directory, file[: -4] + '.jpg')

            boxes, texts = parse_annotation(annotation)
            image = Image.open(image_path)

            for i, box in enumerate(boxes):
                index += 1
                crop_image_path = os.path.join(image_directory, str(index) + '.jpg')
                annotation_path = os.path.join(annotation_directory, str(index) + '.txt')
                all_image_paths.append(crop_image_path)

                crop_image = image.crop(box)
                crop_image.save(crop_image_path)

                with open(annotation_path, 'w') as f:
                    f.write(texts[i])
                    all_texts.append(texts[i])

    with open(os.path.join(image_directory, 'images.json'), 'w') as f:
        json.dump(all_image_paths, f)

    with open(os.path.join(annotation_directory, 'texts.json'), 'w') as f:
        json.dump(all_texts, f)


def create_vocab(annotation_directory='data/task2/annotation'):
    vocab = set()

    for file in os.listdir(annotation_directory):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(os.getcwd(), annotation_directory, file), 'r') as f:
            text = f.read()
            vocab.update(list(text))

    vocab = sorted(vocab)

    with open(os.path.join(os.getcwd(), annotation_directory, 'vocab.json'), 'w') as f:
        json.dump(list(vocab), f)

    return vocab


from src.config import REC_CHAR_SET

# Global caching of character maps for performance
_VOCAB = list(REC_CHAR_SET)
_MAP = {i + 1: char for i, char in enumerate(_VOCAB)}
_REV_MAP = {char: i + 1 for i, char in enumerate(_VOCAB)} 

def encode(text):
    """Encodes a string into a list of integers."""
    text_encode = [_REV_MAP[char] for char in text if char in _REV_MAP]
    return text_encode, len(text_encode)


def decode(labels):
    """
    Decodes a sequence of labels into a string using CTC Greedy Search logic.
    1. Skip if current label is blank (0).
    2. Skip if current label is same as previous (consolidation).
    This correctly handles cases like 'L - blank - L' (L L) vs 'L - L - Blank' (L).
    """
    text = []
    for i in range(len(labels)):
        if labels[i] == 0:
            continue
        if i > 0 and labels[i] == labels[i-1]:
            continue
            
        if labels[i] in _MAP:
            text.append(_MAP[labels[i]])
    
    return ''.join(text)

def create_map():
    """Returns the current maps."""
    return _MAP, _REV_MAP


if __name__ == '__main__':
    print(f"Vocab size: {len(_VOCAB)}")
    print(f"Map size: {len(_MAP)}")
