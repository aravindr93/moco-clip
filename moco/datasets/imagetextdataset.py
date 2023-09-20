import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as transF
import torch.utils.data as data
import webdataset as wds
import webdataset.filters as filters
import struct
import numpy as np
import pdb
import itertools
import torch
import os
from scipy.stats import gamma
from collections import defaultdict
from torchvision.transforms import RandomResizedCrop
from PIL import Image, ImageFilter


class RandomImageTextDataset(data.Dataset):
    """Dummy ImageText dataset that generates random images and texts"""

    def __init__(self, transforms=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        self.vocab = open("/usr/share/dict/words").read().splitlines()

    def __getitem__(self, index):
        rand_img = torch.rand(3, 224, 224)
        word_choice = np.random.choice(len(self.vocab), 3)
        rand_txt = ""
        for i in word_choice:
            rand_txt = rand_txt + self.vocab[i] + " "
        return {"images": rand_img, "label": rand_txt[:-1]}

    def __len__(self):
        return 10000


# =============================
# LAION
# =============================

def blur_faces(img, mask, image_blur_radius, mask_blur_radius):
    blur_img = img.filter(ImageFilter.GaussianBlur(radius=image_blur_radius))
    blur_mask = (
        mask.convert("RGB")
        .filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))
        .convert("1")
    )
    res = Image.composite(blur_img, img, blur_mask)
    return res


def get_face_mask(url, key):
    mask_file = os.path.join(os.path.splitext(url)[0], key + ".jpg-mask.png")
    if os.path.exists(mask_file):
        mask = Image.open(mask_file)

        # Add this part when you have the blur intensity radius.
        radius_bin_file = os.path.join(
            os.path.splitext(url)[0], key + ".jpg-blur-radius.bin"
        )
        # Load the packed data from the file
        with open(radius_bin_file, "rb") as file:
            loaded_data = file.read()
        # Unpack the loaded data into two integers
        unpacked_data = struct.unpack("ff", loaded_data)
        image_blur_radius, mask_blur_radius = unpacked_data

        return mask, image_blur_radius, mask_blur_radius
        # return mask, 5, 2

    return None, None, None


def blur_filter_fn(data, face_masks_folder=None):
    if face_masks_folder is None:
        for sample in data:
            yield sample
    n = 0
    for sample in data:
        if "jpg" not in sample:
            continue
        # print(sample.keys())
        url = os.path.join(
            face_masks_folder,
            sample["__url__"].split("/")[-2],
            sample["__url__"].split("/")[-1],
        )
        key = sample["__key__"]
        mask, image_blur_radius, mask_blur_radius = get_face_mask(url, key)
        if mask is not None:
            sample["jpg"] = blur_faces(
                sample["jpg"], mask, image_blur_radius, mask_blur_radius
            )
        n += 1
        yield sample


blur_filter = filters.pipelinefilter(blur_filter_fn)


def to_dict(data, transform, max_word_len=50):
    for sample in data:
        img = sample[0]   # PIL image
        if transform is not None:
            img = transform(img)
        label = sample[1]["caption"]
        # truncate label to be maximum size with " " as the deliminator
        # note that this doesn't strictly bound the number of tokens
        split_label = label.split(" ")[:max_word_len]
        label = " ".join(split_label)
        yield {
            "images": img,
            "label": label,
        }


class LaionDataset(data.Dataset):
    def __init__(
        self,
        location: str,
        transform=None,
        masks_location=None,
    ):
        data.Dataset.__init__(self)
        # handler = wds.warn_and_continue
        handler = wds.ignore_and_continue
        self.transform = transform
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=handler),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=handler),
            blur_filter(face_masks_folder=masks_location),
            wds.to_tuple("jpg", "json", handler=handler),
            # wds.map(self.to_dict, handler=handler),
            filters.pipelinefilter(to_dict)(transform=self.transform),
        )
        self._iterator = iter(self.inner_dataset)

    def __len__(self):
        # This is a hack. The dataset sampling is not determinitic
        return 100000000

    def __getitem__(self, idx):
        # This is a hack. idx is not being used
        item = next(self._iterator, None)
        if item is None:
            self._iterator = iter(self.inner_dataset)
            item = next(self._iterator, None)
        return item


# =============================
# ImageNet
# =============================

class ImageNetDataset(data.Dataset):
    def __init__(
        self,
        transform=None,
        training_set: bool = True,
        train_val_frac: float = 0.95,
    ):
        data.Dataset.__init__(self)
        self.transform = transform
        self.training_set = training_set
        self.train_val_frac = train_val_frac

        self.labels = {}
        self.label_int = {}
        self.sysnet_labels = {}
        self.all_file_paths = []
        self.load_manifest()

    def load_manifest(self):
        with open("/datasets01/imagenet_full_size/061417/labels.txt", "r") as f:
            int_label = 0
            while True:
                l = f.readline()
                if l == "":
                    break
                key, label = l.split(",")
                label = label.replace("\n", "")
                assert key not in self.labels
                self.labels[key] = label
                self.label_int[key] = int_label
                int_label += 1

        with open(
            "/datasets01/imagenet_full_size/061417/synset_words.txt", "r"
        ) as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                key = l[:9]
                label = l[10:]
                label = label.replace("\n", "")
                assert key not in self.sysnet_labels
                self.sysnet_labels[key] = label

        with open(
            "/checkpoint/maksymets/eaif/datasets/manifests/imagenet_train_manifest.txt",
            "r",
        ) as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                self.all_file_paths.append(l.replace("\n", ""))

    def __len__(self) -> int:
        return len(self.all_file_paths)

    def _get_label_key_from_path(self, path):
        # Path is like /datasets01/imagenet_full_size/061417/train/n01440764/n01440764_10027.JPEG
        l = path.split("/")[-1]
        l = l.split("_")[0]
        return l

    def __getitem__(self, idx):
        """
        Gets one element from the dataset. One element has the keys :
         - images : single image from the ImageNet dataset
         - label : The class of the image
         - sysnet_labels : The labels (comma separated) generated with sysnet
         Example : {"label":"tench", "sysnet_labels":"tench, Tinca tinca" , "images": ...}
        """
        path = self.all_file_paths[idx]
        image = datasets.folder.pil_loader(path)
        label_key = self._get_label_key_from_path(path)
        label = self.labels[label_key]
        int_label = self.label_int[label_key]
        sysnet_labels = self.sysnet_labels[label_key]
        if self.transform is not None:
            image = self.transform(image)
        return {
            "images": image,
            "label": label,
            "sysnet_labels": sysnet_labels,
            "int_label": int_label,
        }


if __name__ == "__main__":
    # LAION test
    print("Testing LAION dataset and loader =====================")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            normalize,
    ])
    dataset = LaionDataset(
        location="/datasets01/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..0}/{00055..00055}.tar",
        masks_location='/checkpoint/haideraltahan/laion440m_masks',
        transform=transforms,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    print(batch.keys())
    print(batch["images"].shape)

    # ImageNet
    print("Testing ImageNet dataset and loader =====================")
    dataset = ImageNetDataset(transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    print(batch.keys())
    print(batch["images"].shape)    
