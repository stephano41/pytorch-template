import logging

from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import NUM_SAMPLES

from utils import instantiate
# from hydra.utils import instantiate


class TrainableOperator(TrainingOperator):
    def setup(self, config):
        logger = logging.getLogger('train')
        # setup data_loader instances
        data_loader, valid_data_loader = instantiate(config.data_loader)

        # build model. print it's structure and # trainable params.
        model = instantiate(config.arch)
        logger.info(model)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

        # get function handles of loss and metrics
        criterion = instantiate(config.loss, is_func=True)
        self.metrics = [instantiate(met, is_func=True) for met in config['metrics']]

        # build optimizer, learning rate tune_scheduler.
        optimizer = instantiate(config.optimizer, model.parameters())
        lr_scheduler = instantiate(config.lr_scheduler, optimizer)
        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(models=model, optimizers=optimizer, criterion=criterion, schedulers=lr_scheduler)
        self.register_data(train_loader=data_loader, validation_loader=valid_data_loader)

    # def train_epoch(self, iterator, info=None, num_steps=None, epoch_idx=0):
    #     self.

    def train_batch(self, batch, batch_info):
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Either set self.model in setup function or "
                "override this method to implement a custom "
                "training loop."
            )
        if not hasattr(self, "optimizer"):
            raise RuntimeError(
                "Either set self.optimizer in setup function "
                "or override this method to implement a custom "
                "training loop."
            )
        if not hasattr(self, "criterion"):
            raise RuntimeError(
                "Either set self.criterion in setup function "
                "or override this method to implement a custom "
                "training loop."
            )
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        *features, target = batch
        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            features = [feature.cuda(non_blocking=True) for feature in features]
            target = target.cuda(non_blocking=True)

        # Compute output.
        with self.timers.record("fwd"):
            if self.use_fp16_native:
                with self._amp.autocast():
                    output = model(*features)
                    loss = criterion(output, target)
            else:
                output = model(*features)
                loss = criterion(output, target)

        # Compute gradients in a backward pass.
        with self.timers.record("grad"):
            optimizer.zero_grad()
            if self.use_fp16_apex:
                with self._amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.use_fp16_native:
                self._amp_scaler.scale(loss).backward()
            else:
                loss.backward()

        # Call step of optimizer to update model params.
        with self.timers.record("apply"):
            if self.use_fp16_native:
                self._amp_scaler.step(optimizer)
                self._amp_scaler.update()
            else:
                optimizer.step()

        batch_size = target.size(0)

        results = {"train_loss": loss.item(), NUM_SAMPLES: batch_size}
        results.update({
            met.__name__: met(output, target) for i, met in enumerate(self.metrics)
        })

        return results

    def validate_batch(self, batch, batch_info):
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Either set self.model in setup function or "
                "override this method to implement a custom "
                "training loop."
            )
        if not hasattr(self, "criterion"):
            raise RuntimeError(
                "Either set self.criterion in setup function "
                "or override this method to implement a custom "
                "training loop."
            )
        model = self.model
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        *features, target = batch
        if self.use_gpu:
            features = [feature.cuda(non_blocking=True) for feature in features]
            target = target.cuda(non_blocking=True)

        # compute output
        with self.timers.record("eval_fwd"):
            if self.use_fp16_native:
                with self._amp.autocast():
                    output = model(*features)
                    loss = criterion(output, target)
            else:
                output = model(*features)
                loss = criterion(output, target)
            # _, predicted = torch.max(output.data, 1)

        num_samples = target.size(0)
        results = { "val_loss": loss.item(),
                    NUM_SAMPLES: num_samples,}

        results.update({
            "val_"+met.__name__: met(output, target) for i, met in enumerate(self.metrics)
        })

        return results

