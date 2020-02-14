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
import argparse
import logging
import os
import sys

import torch
from torch.optim import SGD
from torch.utils.data import ConcatDataset

from dataset_locator import DatasetLocator
from trainer import Trainer

from torch import nn

import torchvision.models as models


class ResnetModel(nn.Module):
    """
    The resnet 50 model, modified to accomodate n classes
    """

    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50()
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        """

        :param x: The input tensor with 3 dimensions
        :return:
        """

        return self.model(x)


class ExperimentWorkflow:
    """
    Experiment workflow to kick off training
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def run(self, dataset_type, train_data_dir, model_dir, batch_size=32, epochs=50,
            num_workers=None, learning_rate=0.0001, device=None):
        # Get list of datasets
        list_dataset = [DatasetLocator().get_datasetfactory(dataset_type, os.path.join(train_data_dir, f)) for f in
                        os.listdir(train_data_dir)]

        num_classes = list_dataset[0].num_classes()

        # Create a concat dataset for all the files..
        self.logger.info("Concatenating {} files".format(len(list_dataset)))
        dataset = ConcatDataset(list_dataset)

        if num_workers is None:
            num_workers = os.cpu_count() - 1

        self.logger.info("Using {} workers".format(num_workers))

        # This is the Dataloader, where the num workers is set
        train_data_batch_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                              num_workers=num_workers)

        # Model
        network = ResnetModel(num_classes=num_classes)

        # Optimiser SGD
        optimiser = SGD(lr=learning_rate, params=network.parameters())

        # Cross entropy loss
        loss_func = nn.CrossEntropyLoss()

        trainer = Trainer()

        trainer.run(dataloader=train_data_batch_loader, network=network, optimizer=optimiser, loss_func=loss_func,
                    num_epochs=epochs, device=device, model_dir=model_dir)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--traindir",
                        help="The input train  dir", default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--dataset_type", help="The input train  dir", type=str, required=True,
                        choices=DatasetLocator().dataset_factory_names)

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))

    parser.add_argument("--modeldir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))

    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)

    parser.add_argument("--batchsize", help="The batch size", type=int, default=32)

    parser.add_argument("--numworkers", help="The number of workers", type=int, default=None)

    parser.add_argument("--lr", help="The learning", type=float, default=0.00001)

    parser.add_argument("--device", help="The device, e.g cpu or cuda:0 for using GPU", type=str, default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args, additional = parser.parse_known_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    workflow = ExperimentWorkflow()

    workflow.run(dataset_type=args.dataset_type, train_data_dir=args.traindir,
                 model_dir=args.modeldir, batch_size=args.batchsize, epochs=args.epochs, num_workers=args.numworkers,
                 device=args.device, learning_rate=args.lr)
