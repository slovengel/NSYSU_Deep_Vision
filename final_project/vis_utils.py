"""Utils for visualizations in notebooks"""
import matplotlib.pyplot as plt
import torch
from math import sqrt, ceil
import numpy as np
from torchvision import transforms
from PIL import Image
from exercise_code.data.segmentation_dataset import label_img_to_rgb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualizer(model, test_data=None):
    num_example_imgs = 4
    plt.figure(figsize=(15, 5 * num_example_imgs))
    for i, (img, target, width, height, img_id) in enumerate(test_data[:num_example_imgs]):
        inputs = img.unsqueeze(0)
        inputs = inputs.to(device)

        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        pred = preds[0].data.cpu()

        img, target, pred = img.numpy(), target.numpy(), pred.numpy()

        # img
        plt.subplot(num_example_imgs, 3, i * 3 + 1)
        plt.axis('off')
        plt.imshow(img.transpose(1, 2, 0))
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 3, i * 3 + 2)
        plt.axis('off')
        plt.imshow(label_img_to_rgb(target))
        if i == 0:
            plt.title("Target image")

        # pred
        plt.subplot(num_example_imgs, 3, i * 3 + 3)
        plt.axis('off')
        plt.imshow(label_img_to_rgb(pred))
        if i == 0:
            plt.title("Prediction image")

    plt.show()

def save_prediction(model, test_data=None):
    for i, (img, target, width, height, img_id) in enumerate(test_data):
        inputs = img.unsqueeze(0)
        inputs = inputs.to(device)

        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        pred = preds[0].data.cpu()
        pred = pred.numpy()

        img = Image.fromarray(label_img_to_rgb(pred))
        center_crop = transforms.CenterCrop([width, height])
        img = center_crop(img)
        img.save('prediction/' + img_id + '.png')

        print(img_id + '.png' + ' saved!')
        