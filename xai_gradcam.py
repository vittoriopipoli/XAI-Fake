from torchcam.methods import LayerCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import transforms
import torch

def plot_activation(model, dataset, path, config):
    i = 0
    cam_extractor = LayerCAM(model, target_layer="conv1")
    for image, target, img_path in dataset:
        image = image.cuda()
        # Preprocess your data and feed it to the model
        out = model(image.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(target.item(), out)
        DeT = transforms.Compose([
            transforms.Normalize(-1 * torch.Tensor(config.data_loader.mean) / torch.Tensor(config.data_loader.std), 1.0 / torch.Tensor(config.data_loader.std))
        ])
        image = DeT(image)
        result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.1)
        #result = image * activation_map[0]
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
        #plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        #plt.savefig('{}/{}Grad_{}.jpg'.format(path, misclass, i))
        plt.show()
        plt.close()

        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title(img_path.split("/")[-1])
        plt.tight_layout()
        plt.show()
        #plt.savefig('{}/{}IMG_{}.jpg'.format(path, misclass, i))
        plt.close()
        i += 1
