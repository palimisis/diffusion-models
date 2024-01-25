from torch.utils.data import Dataset
import os
import natsort
from PIL import Image

class SixRayDataSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        all_imgs = os.listdir(root)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image