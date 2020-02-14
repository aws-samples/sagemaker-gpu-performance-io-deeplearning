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
import logging

import torchvision

from datasets.caltech_image_preprocessor import CaltechImagePreprocessor
from datasets.custom_dataset_base import CustomDatasetBase


class CustomCaltechOptimisedDataset(CustomDatasetBase):

    def __init__(self, file_preprocessed):
        """
Caltech 256, see http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar for more dteails
        :param metadata_file: The metadata file that contains the label names, see https://www.cs.toronto.edu/~kriz/cifar.html for metadata
        :param file: The file to load
        """

        super().__init__()
        # Random crop transformations
        self.transformer = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((224, 224)),
                                                           torchvision.transforms.ToTensor()

                                                           ])

        self.images, self.labels, self.labels_map = CaltechImagePreprocessor().load(file_preprocessed)
        self.logger.debug("Initialised {}".format(dir))

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        curtime = datetime.datetime.now()

        self.logger.debug("{}: retrieving item {}".format(curtime, idx))

        image, label = self.images[idx], self.labels[idx]

        # Apply transformation at each get item
        image = self.transformer(image)

        self.logger.debug("{}: completed item {}".format(datetime.datetime.now(), idx))

        return image, label

    def num_classes(self):
        return 257
