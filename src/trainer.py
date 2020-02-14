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
import os
import time

import torch


class Trainer:
    """
    This class runs the training epochs
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def run(self, dataloader, network, optimizer, loss_func, num_epochs,
            device=None, model_dir=None):

        # Determine device use GPU if available
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info("Using device {}".format(device))

        self.logger.info(
            "Using dataloader with {} batches, the batch size is {} ".format(len(dataloader), dataloader.batch_size))

        # copy network to device [cpu /gpu] if available
        network.to(device=device)

        for epoch in range(num_epochs):
            total_loss = 0
            n_batches = 0
            tic = time.time()

            for i, (features, target) in enumerate(dataloader):
                self.logger.debug("Loaded data for epoch {} : batch {}".format(epoch, i))

                network.train()

                target = target.to(device=device)

                feature_tensor = features.float().to(device=device)

                # Forward pass
                outputs = network(feature_tensor)

                # Calc loss
                loss = loss_func(outputs, target)
                total_loss += loss.item()

                # Zero grad, bp and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.debug("Completed data for batch {}".format(i))
                n_batches = i

            # This is for Sagemaker plotting
            print("## loss ##: {}".format(total_loss))
            print("## secs_time_per_epoch ##: {}".format(time.time() - tic))

            self.logger.info('Epoch [%d/%d], total loss: %.4f, avg loss: %.4f, timecost: %d'
                             % (epoch + 1, num_epochs, total_loss, total_loss / n_batches, time.time() - tic))

        if model_dir is not None:
            self.logger.info("Saving model ..")
            snapshot_path = os.path.join(model_dir, "model.pth")
            torch.save(network.state_dict(), snapshot_path)
