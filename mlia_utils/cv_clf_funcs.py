import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

def transform_imgs(p=0.5):
  return torchvision.transforms.Compose([
      transforms.Resize((150,150)),
      transforms.RandomHorizontalFlip(p=p), # randomly flip and rotate
      transforms.ColorJitter(0.3,0.4,0.4,0.2),
      transforms.ToTensor(),
      transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
      ])