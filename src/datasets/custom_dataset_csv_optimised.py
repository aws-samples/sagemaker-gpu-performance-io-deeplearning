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

import logging

import pandas as pd

from datasets.custom_dataset_base import CustomDatasetBase


class CustomDatasetCsvOptimised(CustomDatasetBase):

    def __init__(self, file):
        """
This improves the performance of :class:`DataLoaderCsv` by improving the __getitem__ function
        :param file:
        """
        super().__init__()
        self.data = self._load_data(file)

    @staticmethod
    def _load_data(file):
        logging.debug("Loading entire file {} into memory".format(file))

        # Preprocess into numpy to avoid repeated numpy array convert on get item for each row
        data = pd.read_csv(file, header=None).values

        logging.debug("Completed entire file {} into memory".format(file))

        return data

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Data is fully preprocessed and ready to go making it much faster
        return self.data[idx, 0:-1], self.data[idx, -1]

    def num_classes(self):
        return 10
