from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


def plot_activation(model, dataset, path):
    i = 0
    cam_extractor = SmoothGradCAMpp(model)
    for image, target, img_path in dataset:
        image = image.cuda()
        # Preprocess your data and feed it to the model
        out = model(image.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(target.item(), out)
        result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # Display it
        misclass = ""
        real=""
        if out[0].argmax().item() != target:
            misclass = "misclass_"
        if target == 1:
            real = "fake_"
        misclass += real
        plt.imshow(result)
        plt.axis('off')
        plt.title(img_path.split("/")[-1])
        plt.tight_layout()
        plt.savefig('{}/{}Grad_{}.jpg'.format(path, misclass, i))
        plt.close()

        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(img_path.split("/")[-1])
        plt.tight_layout()
        plt.savefig('{}/{}IMG_{}.jpg'.format(path, misclass, i))
        plt.close()
        i += 1
