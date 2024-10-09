# /workspace/public/users/lichang93/mydocker/cogvx/host_folder/Open-Sora-Plan/mytest/dataloader_test.py(212)main()
# -> train_dataloader = accelerator.prepare(train_dataloader)
#   /root/mambaforge/lib/python3.10/site-packages/accelerate/accelerator.py(1303)prepare()
# -> result = self._prepare_deepspeed(*args)
#   /root/mambaforge/lib/python3.10/site-packages/accelerate/accelerator.py(1597)_prepare_deepspeed()
# -> result = [
#   /root/mambaforge/lib/python3.10/site-packages/accelerate/accelerator.py(1598)<listcomp>()
# -> self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
#   /root/mambaforge/lib/python3.10/site-packages/accelerate/accelerator.py(1186)_prepare_one()
# -> return self.prepare_data_loader(obj, device_placement=device_placement)
# > /root/mambaforge/lib/python3.10/site-packages/accelerate/accelerator.py(2028)prepare_data_loader()
# -> prepared_data_loader = prepare_data_loader(
#   /root/mambaforge/lib/python3.10/site-packages/accelerate/data_loader.py(921)prepare_data_loader()
# -> dist.breakpoint(0)

# %% accelerator.py(2028)prepare_data_loader()
# dataloader.batch_size : 1
{'_custom_objects': [],
 '_dataloaders': [],
 '_load_model_state_pre_hook': OrderedDict(),
 '_models': [],
 '_optimizers': [],
 '_save_model_state_pre_hook': OrderedDict(),
 '_schedulers': [],
 'autocast_handler': None,
 'dataloader_config': DataLoaderConfiguration(split_batches=False,
                                              dispatch_batches=None,
                                              even_batches=True,
                                              use_seedable_sampler=False,
                                              non_blocking=False),
 'ddp_handler': None,
 'delayed_fp8_autocast': False,
 'device_placement': True,
 'flag_tensor': None,
 'fp8_recipe_handler': None,
 'gradient_state': Sync Gradients: True
At end of current dataloader: False
Extra samples added: -1
Gradient accumulation plugin: {'num_steps': 1}
,
 'has_lomo_optimizer': False,
 'init_handler': None,
 'log_with': [<LoggerType.WANDB: 'wandb'>],
 'native_amp': False,
 'profile_handler': None,
 'project_configuration': ProjectConfiguration(project_dir='/workspace/Open-Sora-Plan/runs/test/',
                                               logging_dir=PosixPath('/workspace/Open-Sora-Plan/runs/test/logs'),
                                               automatic_checkpoint_naming=False,
                                               total_limit=None,
                                               iteration=0,
                                               save_on_each_node=False),
 'rng_types': ['generator'],
 'scaler': None,
 'scaler_handler': None,
 'state': Distributed environment: DEEPSPEED  Backend: nccl
Num processes: 8
Process index: 0
Local process index: 0
Device: cuda:0

Mixed precision type: bf16
ds_config: {'fp16': {'enabled': False, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 16, 'hysteresis': 2, 'min_loss_scale': 1}, 'bf16': {'enabled': True}, 'communication_data_type': 'fp32', 'gradient_clipping': 1.0, 'train_micro_batch_size_per_gpu': 'auto', 'train_batch_size': 'auto', 'gradient_accumulation_steps': 'auto', 'zero_optimization': {'stage': 2, 'overlap_comm': True, 'contiguous_gradients': True, 'sub_group_size': 1000000000.0, 'reduce_bucket_size': 500000000.0}, 'steps_per_print': inf}
,
 'step': 0,
 'step_scheduler_with_optimizer': True,
 'trackers': []}

# 2028         prepared_data_loader = prepare_data_loader(
# 2029             data_loader,
# 2030             self.device,
# 2031             num_processes=self.num_processes,
# 2032             process_index=self.process_index,
# 2033             split_batches=self.split_batches,
# 2034             put_on_device=device_placement,
# 2035             rng_types=self.rng_types.copy(),
# 2036             dispatch_batches=self.dispatch_batches,
# 2037             even_batches=self.even_batches,
# 2038             slice_fn_for_dispatch=slice_fn_for_dispatch,
# 2039             use_seedable_sampler=self.use_seedable_sampler,
# 2040             non_blocking=self.non_blocking,
# 2041         )



# %% BatchSamplerShard(BatchSampler):

# each self.process_index will only yield
# `idx % self.num_processes == self.process_index`th batch
#
def _iter_with_no_split(self):
    initial_data = []
    batch_to_yield = []
    for idx, batch in enumerate(self.batch_sampler):
        # We gather the initial indices in case we need to circle back at the end.
        if not self.drop_last and idx < self.num_processes:
            initial_data += batch
        # We identify the batch to yield but wait until we ar sure every process gets a full batch before actually
        # yielding it.
        if idx % self.num_processes == self.process_index:
            batch_to_yield = batch
        if idx % self.num_processes == self.num_processes - 1 and (
            self.batch_size is None or len(batch) == self.batch_size
        ):
            yield batch_to_yield
            batch_to_yield = []


# %% data_loader.py(921)prepare_data_loader()

        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != DistributedType.TPU else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )

# (Pdb) pp kwargs
# rng_types: ['generator']
{'collate_fn': <opensora.utils.dataset_utils.Collate object at 0x7fef54b43610>,
 'generator': None,
 'multiprocessing_context': None,
 'num_workers': 0,
 'persistent_workers': False,
 'pin_memory': False,
 'prefetch_factor': None,
 'timeout': 0,
 'worker_init_fn': None}

# %% Reproducible Sampler: data_loader.py(921)prepare_data_loader()
# 864         use_seedable_sampler (`bool`, *optional*, defaults to `False`):
# 865             Whether to use the [`~data_loader.SeedableRandomSampler`] instead of a `RandomSampler` for better
# 866             reproducability. Comes at a cost of potentially different performances due to different shuffling
# 867             algorithms but ensures results will be the *exact* same. Should be paired with `set_seed()` at every
# 868             `self.set_epoch`


# If Not Random Sampler:
# 1. all ranks: a torch.Generator created in `data_loader.py(694)prepare_data_loader()`
            if hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator()
                synchronized_generator = sampler.generator

# 2. sync (broadcast rank 0) all generators:
    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)

# ??? If Random Sampler: ??? How every rank uses the same seed?
    if isinstance(sampler, RandomSampler) and use_seedable_sampler:
        # When iterating through the dataloader during distributed processes
        # we want to ensure that on each process we are iterating through the same
        # samples in the same order if a seed is set. This requires a tweak
        # to the `torch.utils.data.RandomSampler` class (if used).
        sampler = SeedableRandomSampler(
            data_source=sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler._num_samples,
            generator=getattr(sampler, "generator", torch.Generator()),
        )


    class SeedableRandomSampler(RandomSampler):
        """
        Same as a random sampler, except that in `__iter__` a seed can be used.
        """
        def __iter__(self):
            if self.generator is None:
                self.generator = torch.Generator()
                self.generator.manual_seed(self.initial_seed)

            # Allow `self.epoch` to modify the seed of the generator
            seed = self.epoch + self.initial_seed
            # print("Setting seed at epoch", self.epoch, seed)
            self.generator.manual_seed(seed)
