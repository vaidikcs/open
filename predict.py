from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import torch
from torchvision.transforms import Compose
from cog import BasePredictor, Input, Path
import numpy as np
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.encoder_large = 'vitl' # can also be 'vitb' or 'vitl'
        self.encoder_small = 'vits'
        self.large_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(self.encoder_large)).eval()
        print("large model loaded.")
        self.small_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(self.encoder_small)).eval()
        print("both model loaded.")


        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    
    def predict(
        self,
        image: Path = Input(
            description=f"input image"),
        model: str = Input(
            description="type of model",
            default='small'
        )
        ) -> Path:
        with Image.open(image) as img:
            # Convert the image to a NumPy array
            image = np.array(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({'image': image})['image']
        # image.save('1-processed.jpg')
        image = torch.from_numpy(image).unsqueeze(0)

        # depth shape: 1xHxW
        if model=='small':
            inv_depth = self.small_model(image)
        else:
            inv_depth = self.large_model(image)

        inv_depth = inv_depth.cpu().detach().numpy().tobytes()

        return Path(inv_depth)
