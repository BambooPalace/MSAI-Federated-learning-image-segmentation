import os

import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision import transforms
from PIL import Image

from options import args_parser
from federated_main import make_model
from coco_utils import coco_classes


def inference(model, checkpoint, filename, root, num_classes):
    path = os.path.join(root, checkpoint)
    checkpoint = torch.load(path, map_location='cpu')
    print(f'Loading check point from experiment :' + checkpoint['exp_name'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    if not filename:
        exit('Please input filename for inference.')
    file_path = os.path.join(root, filename)
    input_image = Image.open(file_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    print('Inferencing on image.')
    output = model(input_batch)['out'][0]
    predictions = output.argmax(0)
    classes = torch.unique(predictions).tolist()
    # Class IDs are remapped so need to converted back
    catIds = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
    classes = [catIds[c] for c in classes]    
    classes = [coco_classes[c] for c in classes if c != 0]
    print(f'Predicted classes are {classes}')

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 31 - 1])    
    colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    # ax.imshow(r)
    path = os.path.join(root, '21', filename.split('.')[0] + '_mask.jpg')
    r.convert('RGB').save(path)
    print('Segmentation mask saved.')

if __name__ == '__main__':
    args = args_parser()    
    model = make_model(args)
    inference(model, args.checkpoint, args.filename, args.root, args.num_classes)

    
#command line
'''
python segmentation/inference.py --activation=relu --root=./test --checkpoint=checkpoint4-relu.pth --filename=image1.jpg 
python segmentation/inference.py --activation=tanh --root=./test --checkpoint=checkpoint31-dp-tanh.pth --dp --filename=image1.jpg
'''