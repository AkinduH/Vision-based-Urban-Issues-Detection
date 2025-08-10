# utils/dataset.py
import math
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')

class Dataset(data.Dataset):
    """
    YOLO-style dataset loader WITHOUT augmentations.
    Expects filenames: a list of image absolute paths.
    Corresponding labels must be in labels/... with same filename stem and
    each line: class x_center y_center width height  (normalized 0..1)
    """
    def __init__(self, filenames, input_size, params=None, augment=False):
        self.params = params or {}
        self.input_size = input_size
        # We do NOT use augmentation for this version
        self.augment = False

        # Read labels (returns dict: image_path -> ndarray (n,5))
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())
        self.n = len(self.filenames)
        self.indices = range(self.n)

    def __getitem__(self, index):
        index = self.indices[index]

        # Load image
        image, shape = self.load_image(index)
        h, w = image.shape[:2]

        # Resize (no augmentation)
        image, ratio, pad = resize(image, self.input_size, augment=False)

        # Load label (already cached & in normalized xywh; convert to pixel xyxy)
        label = self.labels[index].copy()
        if label.size:
            # convert from normalized xywh to absolute x1y1x2y2 w.r.t resized+padded image
            label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
        else:
            label = np.zeros((0, 5), dtype=np.float32)

        nl = len(label)
        h_resized, w_resized = image.shape[:2]

        # cls: (n,1), box: (n,4) in normalized xywh w.r.t resized image
        cls = label[:, 0:1] if nl else np.zeros((0, 1), dtype=np.float32)
        box = label[:, 1:5] if nl else np.zeros((0, 4), dtype=np.float32)
        box = xy2wh(box, w_resized, h_resized)  # convert to normalized xywh relative to resized image

        # Ensure target tensors shapes are consistent
        target_cls = torch.zeros((nl, 1), dtype=torch.float32)
        target_box = torch.zeros((nl, 4), dtype=torch.float32)
        if nl:
            target_cls = torch.from_numpy(cls).float().view(-1, 1)
            target_box = torch.from_numpy(box).float().view(-1, 4)

        # Convert HWC BGR -> CHW RGB (and contiguous)
        sample = image[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR->RGB and HWC->CHW
        sample = np.ascontiguousarray(sample, dtype=np.float32) / 255.0
        sample = torch.from_numpy(sample)

        # idx placeholder (used by your ComputeLoss pipeline)
        idx = torch.zeros(nl, dtype=torch.long)

        return sample, target_cls, target_box, idx

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        if image is None:
            raise FileNotFoundError(self.filenames[i])
        h, w = image.shape[:2]
        return image, (h, w)

    @staticmethod
    def collate_fn(batch):
        """
        Robust collate that normalizes cls to (n,1) and box to (n,4)
        Returns: imgs tensor (B,C,H,W) and targets dict {'cls','box','idx'}
        where cls:(N,1), box:(N,4), idx:(N,) with idx pointing to image index in batch
        """
        samples, cls_list, box_list, indices = zip(*batch)

        imgs = torch.stack(samples, dim=0)

        # normalize cls shapes to (n,1)
        cls_fixed = []
        for c in cls_list:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c)
            if c.numel() == 0:
                c = c.reshape(0, 1)
            elif c.dim() == 1:
                c = c.view(-1, 1)
            elif c.dim() == 2:
                # ensure shape (n,1) or (n,k) -> keep as-is (expected (n,1))
                pass
            else:
                c = c.view(-1, 1)
            cls_fixed.append(c.float())

        # normalize box shapes to (n,4)
        box_fixed = []
        for b in box_list:
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b)
            if b.numel() == 0:
                b = b.reshape(0, 4)
            elif b.dim() == 1:
                if b.numel() == 4:
                    b = b.view(1, 4)
                else:
                    b = b.view(-1, 4)
            elif b.dim() == 2:
                pass
            else:
                b = b.view(-1, 4)
            box_fixed.append(b.float())

        # concat or create empty tensors
        cls = torch.cat(cls_fixed, dim=0) if len(cls_fixed) else torch.zeros((0, 1))
        box = torch.cat(box_fixed, dim=0) if len(box_fixed) else torch.zeros((0, 4))

        # build idx: add batch offsets so idx refers to which image each target belongs to
        new_indices = []
        for i, ind in enumerate(indices):
            if not isinstance(ind, torch.Tensor):
                ind = torch.tensor(ind)
            if ind.numel() == 0:
                new_indices.append(ind.reshape(0))
            else:
                new_indices.append(ind + i)
        idx = torch.cat(new_indices, dim=0) if any(n.numel() for n in new_indices) else torch.zeros((0,), dtype=torch.long)

        targets = {'cls': cls, 'box': box, 'idx': idx}
        return imgs, targets

    @staticmethod
    def load_label(filenames):
        """
        filenames: list of image absolute paths
        returns dict {image_path: ndarray(n,5)} where each row is [class,x,y,w,h] (float32)
        This parser is tolerant: it will take the first 5 numeric tokens per line and ignore extras.
        """
        cache_path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        x = {}
        for filename in filenames:
            try:
                # verify image
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                if not ((shape[0] > 9) and (shape[1] > 9)):
                    # skip tiny images
                    continue
                if image.format is None or image.format.lower() not in FORMATS:
                    continue

                # label path: replace '/images/' with '/labels/' and change ext to .txt
                a = os.sep + 'images' + os.sep
                b = os.sep + 'labels' + os.sep
                label_path = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(label_path):
                    good = []
                    with open(label_path, 'r') as lf:
                        for line in lf.read().strip().splitlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # keep first 5 values, ensure numeric
                                try:
                                    vals = [float(x) for x in parts[:5]]
                                    good.append(vals)
                                except:
                                    # skip non-numeric line
                                    continue
                    if len(good):
                        label = np.array(good, dtype=np.float32)
                        # basic assertions
                        if label.shape[1] != 5:
                            label = label[:, :5]
                        # ensure normalized coords in [0,1] for x,y,w,h
                        if not ((label[:, 1:] <= 1).all() and (label[:, 1:] >= 0).all()):
                            # if some labels are outside 0..1, clip them (safe fallback)
                            label[:, 1:] = np.clip(label[:, 1:], 0.0, 1.0)
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)

            except FileNotFoundError:
                label = np.zeros((0, 5), dtype=np.float32)
            except AssertionError:
                continue

            x[filename] = label

        # save cache
        torch.save(x, cache_path)
        return x


# --- helper functions (resize, wh2xy, xy2wh, etc.) ---
def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y

def xy2wh(x, w, h):
    if x.size == 0:
        return x.reshape((0,4))
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y

def resample():
    choices = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR,
               cv2.INTER_NEAREST, cv2.INTER_LANCZOS4)
    return random.choice(choices)

def resize(image, input_size, augment):
    # Resize and pad to square input_size
    shape = image.shape[:2]  # h, w
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:
        r = min(r, 1.0)
    new_w = int(round(shape[1] * r))
    new_h = int(round(shape[0] * r))
    if (new_w, new_h) != (shape[1], shape[0]):
        image = cv2.resize(image, dsize=(new_w, new_h), interpolation=resample() if augment else cv2.INTER_LINEAR)
    pad_w = (input_size - new_w) / 2
    pad_h = (input_size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return image, (r, r), (left, top)
