# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import datetime
import glob
import logging
import os

import torchvision
from PIL import Image

from datasets.custom_dataset_base import CustomDatasetBase


class CustomCaltechDataset(CustomDatasetBase):

    def __init__(self, dir):
        """
Caltech 256, see http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar for more dteails
        :param metadata_file: The metadata file that contains the label names, see https://www.cs.toronto.edu/~kriz/cifar.html for metadata
        :param file: The file to load
        """

        super().__init__()
        self.transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                           torchvision.transforms.RandomCrop((224, 224)),
                                                           torchvision.transforms.ToTensor()

                                                           ])

        self.images, self.labels, self.labels_map = self._load_images(dir)
        self.logger.debug("Initialised {}".format(dir))

    @staticmethod
    def _load_images(dir):
        logging.debug("Loading dir {} ".format(dir))
        extn = ".jpg"
        labels_map = {}

        # Load dir
        images, labels = [], []
        for f in glob.glob("{}/**/*{}".format(dir, extn)):
            # file name format convention , e.g 001.ak47/001_0001.jpg
            file_name = os.path.basename(f).rstrip(extn)
            label = int(file_name.split("_")[0]) - 1

            # Get the label name based on the directory
            label_name = os.path.dirname(f).split(os.path.sep)[-1].split(".")[1]

            images.append(f)
            labels.append(label)

            labels_map[label] = label_name

        logging.info("Completed loading {}. Found {} files, with {} labels".format(dir, len(images), len(labels_map)))

        # Return data frame
        return images, labels, labels_map

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        curtime = datetime.datetime.now()

        self.logger.debug("{}: retrieving item {}".format(curtime, idx))

        image, label = self.images[idx], self.labels[idx]

        # Convert to PIL image to apply transformations
        image = Image.open(image)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')

        # Apply transformation at each get item
        image = self.transformer(image)
        self.logger.debug("{}: completed item {}".format(datetime.datetime.now(), idx))

        return image, label

    def num_classes(self):
        return 257
