from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

def plot_activation(model, dataset, ):
    for image in dataset:
        with SmoothGradCAMpp(model) as cam_extractor:
          # Preprocess your data and feed it to the model
          out = model(image.unsqueeze(0))
          # Retrieve the CAM by passing the class index and the model output
          activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
          result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
          # Display it
          plt.imshow(result);plt.axis('off');plt.tight_layout();plt.show()
