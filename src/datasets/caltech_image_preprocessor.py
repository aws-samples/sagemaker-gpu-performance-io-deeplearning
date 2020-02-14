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
import pickle

import torchvision
from PIL import Image


class CaltechImagePreprocessor:
    """
    Preprocess the Caltech images to a PIL image that is pickled
    """

    def dump(self, images_base_directory, destination_pickle_path, resize=(256, 256), parts=4):
        """
Pickles the raw images
        :param parts: Divides the pickle into n parts, with number of images divided almost equally per part
        :param resize: The resize spec H , W
        :param images_base_directory: Expected format file/dir name convention, e.g 001.ak47/001_0001.jpg

        :param destination_pickle_path:
        """
        logging.debug("Loading images_base_directory {} ".format(images_base_directory))
        extn = ".jpg"
        labels_map = {}
        transform_resize = torchvision.transforms.Resize(resize)

        # Load images_base_directory
        images, labels = [], []

        images_list = list(glob.glob("{}/**/*{}".format(images_base_directory, extn), recursive=True))

        files_per_pickle = len(images_list) // parts
        pickle_part_num = 1
        result = None
        for i, f in enumerate(images_list):

            # file name format convention , e.g 001.ak47/001_0001.jpg
            file_name = os.path.basename(f).rstrip(extn)
            label = int(file_name.split("_")[0]) - 1

            # Get the label name based on the directory
            label_name = os.path.dirname(f).split(os.path.sep)[-1].split(".")[1]

            # Convert to PIL image as part of data loading..
            image = Image.open(f)
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')

            # Resize image
            image = transform_resize(image)

            images.append(image)
            labels.append(label)

            labels_map[label] = label_name

            result = {
                "images": images,
                "labels": labels,
                "labels_map": labels_map
            }

            if (i + 1) % files_per_pickle == 0:
                # Write to file

                self._save_part(destination_pickle_path, pickle_part_num, result)

                # Reset images & labels array
                images = []
                labels = []
                result = None

                pickle_part_num += 1

        # Save final remaining parts
        self._save_part(destination_pickle_path, pickle_part_num, result)

        print("{} Pickled to {}, created {} parts ".format(datetime.datetime.now(), destination_pickle_path,
                                                           pickle_part_num))

    def _save_part(self, destination_pickle_path, pickle_part, result_obj_to_pickle):
        if result_obj_to_pickle is None: return

        curtime = datetime.datetime.now()

        print("{} : Processed {} images with {} labels, pickling part {}".format(curtime,
                                                                                 len(result_obj_to_pickle["images"]),
                                                                                 len(result_obj_to_pickle[
                                                                                         "labels_map"]), pickle_part))

        pickle_file_path = os.path.join(destination_pickle_path, "images_pickle_{}.pkl".format(pickle_part))
        with open(pickle_file_path, "wb") as f:
            pickle.dump(result_obj_to_pickle, f)

    def load(self, pickle_path):
        """
        Load pickled file
        :param pickle_path:
        :return:
        """
        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)

        return obj["images"], obj["labels"], obj["labels_map"]
