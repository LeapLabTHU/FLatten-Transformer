import matplotlib.pyplot as plt
from torchvision import transforms
from models import build_model
from config import get_config
from PIL import Image
import numpy as np
import matplotlib
import argparse
import torch
import tqdm
import os


class AttnVisualizer:
    def __init__(self, qk=None, attn=None, kernel=None, name=''):
        assert (qk is not None and attn is None) or (qk is None and attn is not None)

        # softmax attention
        self.attn = attn

        # linear attention
        if qk is not None:
            self.attn = qk[0] @ qk[1].transpose(-2, -1)
            self.attn = self.attn / self.attn.sum(dim=-1, keepdim=True)

        if len(self.attn.shape) == 4:
            self.attn = self.attn[:, 0, :, :]
        self.kernel = kernel
        self.name = name

        os.makedirs('./visualize', exist_ok=True)

    @staticmethod
    def set_flag(path, flag):
        with open(os.path.join(path, 'flag.txt'), mode='w') as f:
            f.write(str(flag))

    @staticmethod
    def get_flag(path):
        if not os.path.exists(os.path.join(path, 'flag.txt')):
            flag = 0
        else:
            with open(os.path.join(path, 'flag.txt'), mode='r') as f:
                flag = int(f.readlines()[-1])
        return flag

    def get_attn_matrix(self):
        attn_eq = self.attn[0, :, :].clone()

        # add dwc kernel if given
        if self.kernel is not None:
            kernel = self.kernel[0, 0, :, :].clone()
            a = int(attn_eq.shape[0] ** 0.5)
            n = int((kernel.shape[0] - 1) / 2)
            conv_mask = torch.zeros(size=(attn_eq.shape[0], attn_eq.shape[1] + n * (a + 1) * 2))
            for i in range(attn_eq.shape[0]):
                for j in range(kernel.shape[0]):
                    conv_mask[i, i + j * a:i + j * a + kernel.shape[1]] = kernel[j]
            conv_mask = conv_mask[:, n * (a + 1):n * (a + 1) + attn_eq.shape[1]]
            attn_eq = attn_eq + conv_mask

            # visualize the absolute value of equivalent attention
            # because dwc kernel could be negative
            attn_eq = torch.abs(attn_eq)

        # normalize
        attn_eq = attn_eq / attn_eq.sum(dim=-1, keepdim=True)
        # multiply by (attn_eq.shape[0] / 196) * 10 for better visualization
        attn_eq = attn_eq * (attn_eq.shape[0] / 196) * 10
        attn_eq[attn_eq > 1] = 1

        return attn_eq

    def visualize_all_attn(self, max_num=None, image=None, **kwargs):
        path = './visualize/' + self.name + '_all'
        if not os.path.exists(path):
            os.mkdir(path)
        all_attn = self.get_all_attn(max_num=max_num, **kwargs)
        flag = self.get_flag(path=path)
        count = flag
        if not os.path.exists(path + '/' + self.name + '_' + str(count)):
            os.mkdir(path + '/' + self.name + '_' + str(count))
        if image is None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            for i in tqdm.tqdm(range(len(all_attn))):
                plt.matshow(all_attn[i], cmap='Blues', norm=norm)
                plt.colorbar()
                plt.title('Attention Mask')
                plt.savefig(path + '/' + self.name + '_' + str(count) + '/' + str(i) + '.png', dpi=600)
                plt.close()
        else:
            image = np.array(Image.open(image))
            for i in tqdm.tqdm(range(len(all_attn))):
                result = self.mask_image(image, all_attn[i])
                result.save(path + '/' + self.name + '_' + str(count) + '/' + str(i) + '.png')
            if count == 0:
                if count == 0:
                    n = all_attn[0].shape[0] * all_attn[0].shape[1]
                    sep = 1
                    if max_num is not None:
                        import math
                        sep = math.ceil(n / max_num)
                        n = n // sep
                    if not os.path.exists(path + '/query'):
                        os.mkdir(path + '/query')
                    for i in range(n):
                        attn = np.zeros(shape=(all_attn[0].shape[0], all_attn[0].shape[1]), dtype=float)
                        attn[(i * sep) // all_attn[0].shape[1], (i * sep) % all_attn[0].shape[1]] = 1.0
                        result = self.mask_image(image, attn, alpha=-1, color=[255., 33., 33.])
                        result.save(path + '/query/' + str(i) + '.png')
        self.set_flag(path=path, flag=flag + 1)

    def get_all_attn(self, max_num=None):
        attn = self.get_attn_matrix()
        remain = attn.shape[1] - int(int(attn.shape[1] ** 0.5) ** 2)
        n = attn.shape[0]
        m = attn.shape[1] - remain
        shape = [int(m ** 0.5), int(m ** 0.5)]
        if max_num is not None:
            import math
            sep = math.ceil(n / max_num)
            n = n // sep
        all_attn = []
        for i in range(n):
            if max_num is None:
                temp = attn[i, remain:]
            else:
                temp = attn[i * sep, remain:]
            temp = temp.reshape(shape[0], shape[1]).cpu()
            temp_numpy = temp.numpy()
            all_attn.append(temp_numpy)
        return all_attn

    @staticmethod
    def mask_image(image, attn, color=None, alpha=0.3):
        background = [224., 224., 224.]
        k = 1.0
        attn = attn ** k
        image = image.astype(float)
        attn = torch.tensor(attn).unsqueeze(dim=0).unsqueeze(dim=0)
        attn = torch.nn.functional.interpolate(attn, size=(image.shape[0], image.shape[1]), mode='nearest')
        attn = attn.squeeze(dim=0).squeeze(dim=0).unsqueeze(dim=-1).numpy()
        if color is None:
            color = [0., 0., 255.]
        background = np.array(background).reshape(1, 1, 3)
        color = np.array(color).reshape(1, 1, 3)
        if alpha > 0:
            image = image * alpha + (attn * color + (1 - attn) * background) * (1 - alpha)
        else:
            image = image * (1 - attn) + attn * color
        image = image.astype(np.uint8)
        return Image.fromarray(image)


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='./cfgs/flatten_swin_t.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=[],
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    _, config = parse_option()
    model = build_model(config)
    model.load_state_dict(torch.load('flatten_swin_t_pretrained.pth')['model'])
    model.eval()
    image = Image.open('./visualize/img_ori_00809.png')
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = t(image).reshape(1, 3, 224, 224)
    with torch.no_grad():
        y = model(image)


