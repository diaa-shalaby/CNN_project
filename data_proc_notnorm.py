from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
import torch
from crop import crop_image

DATA_PATH = Path("/Excercise_5\data_set")


class Image_dataset(Dataset):
    def __init__(self, start_seq, end_seq):
        self.img_arrays = []

        # It makes sense to only work with inputs of sizes that can appear in the test set.
        im_shape = 90
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape)),
            # transforms.ToTensor(dtype=torch.uint8)
        ])

        for i in range(start_seq, end_seq):
            i = '{:0>3}'.format(i)
            folder = os.path.join(DATA_PATH, i)
            for image_file in os.listdir(folder):
                with Image.open(Path(folder, image_file)) as im:
                    im = resize_transforms(im)
                    img = np.asarray(im, dtype=np.uint8)
                    self.img_arrays.append(img)

    def random_within_range(self):
        start = np.random.randint(5, 10)
        new_max = 15 - start
        end = np.random.randint(5, new_max)

        return start, end

    def __len__(self):
        return len(self.img_arrays)

    def __getitem__(self, idx):
        img_array = self.img_arrays[idx]

        border_x = self.random_within_range()
        border_y = self.random_within_range()

        img_arrays, known_arrs, target_arrs = crop_image(img_array, border_x, border_y)

        return img_arrays, known_arrs, target_arrs, idx


def stack_images_arrays(batch_as_list):
    n_samples = len(batch_as_list)
    n_features = 2
    img_arrays = [batch[0] for batch in batch_as_list]

    known_arrs = [batch[1] for batch in batch_as_list]

    target_arrs = [batch[2] for batch in batch_as_list]

    max_X = np.max([img_array.shape[0] for img_array in img_arrays])
    max_y = np.max([img_array.shape[1] for img_array in img_arrays])

    stacked_imgs_knowns = torch.zeros(size=(n_samples, n_features,
                                            max_X, max_y), )

    for i, img_arr in enumerate(img_arrays):
        stacked_imgs_knowns[i, 0, :max_X, :max_y] = torch.tensor(img_arr)

    for i, known_arr in enumerate(known_arrs):
        stacked_imgs_knowns[i, 1, :max_X, :max_y] = torch.tensor(known_arr)

    stacked_targets = torch.zeros(size=(n_samples, 2475))

    for i, target in enumerate(target_arrs):
        stacked_targets[i, :int(target.shape[0])] = torch.tensor(target)

    # target_tensors = torch.as_tensor([target for target in target_arrs])

    labels = [batch_label[3] for batch_label in batch_as_list]

    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.uint8) for label in labels], dim=0)

    return stacked_imgs_knowns, stacked_targets, stacked_labels


class Test_set(Dataset):
    def __init__(self, testset):
        self.img_arrays = []
        self.borders_x = []
        self.borders_y = []
        for idx in range(208):
            img_array = testset["input_arrays"][idx]
            border_x = testset['borders_x'][idx]
            border_y = testset['borders_y'][idx]

            self.img_arrays.append(img_array)
            self.borders_x.append(border_x)
            self.borders_y.append(border_y)

    def __len__(self):
        return len(self.img_arrays)

    def __getitem__(self, idx):
        img_arr = self.img_arrays[idx]
        b_x = self.borders_x[idx]
        b_y = self.borders_y[idx]

        img_array, known_array, target_array = crop_image(img_arr, b_x, b_y)

        return img_array, known_array, target_array, idx




