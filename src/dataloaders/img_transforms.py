from torchvision.transforms import *
from PIL import Image
import random
import math

class ResizeWithEqualScale(object):
    """
    Resize an image with equal scale as the original image.
    Args:
        height (int): resized height.
        width (int): resized width.
        interpolation: interpolation manner.
        fill_color (tuple): color for padding.
    """
    def __init__(self, height, width, interpolation=Image.BILINEAR, fill_color=(0,0,0)):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        width, height = img.size
        if self.height / self.width >= height / width:
            height = int(self.width * (height / width))
            width = self.width
        else:
            width = int(self.height * (width / height))
            height = self.height 

        resized_img = img.resize((width, height), self.interpolation)
        new_img = Image.new('RGB', (self.width, self.height), self.fill_color)
        new_img.paste(resized_img, (int((self.width - width) / 2), int((self.height - height) / 2)))

        return new_img


class RandomCroping(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, p=0.5, interpolation=Image.BILINEAR):
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size
        if random.uniform(0, 1) >= self.p:
            return img
        
        new_width, new_height = int(round(width * 1.125)), int(round(height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - width
        y_maxrange = new_height - height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + width, y1 + height))

        return croped_img


class RandomErasing(object):
    """ 
    Randomly selects a rectangle region in an image and erases its pixels.
    Reference:
        Zhong et al. Random Erasing Data Augmentation. arxiv: 1708.04896, 2017.
    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

