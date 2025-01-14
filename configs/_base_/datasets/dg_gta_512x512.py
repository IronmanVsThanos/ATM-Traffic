_base_ = [
    # "./RS3K_512x512.py",
    # "./snow-acdc_1024x1024.py",
    # "./rain-acdc_1024x1024.py",
    # "./fog-acdc_1024x1024.py",
    # "./night-acdc_1024x1024.py",
    "./tsp6k_1024x1024.py",
    # "./tsp6k_512x512.py",
    ]
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    # dataset={{_base_.train_gta}},
    dataset=dict(
            type="ConcatDataset",
            datasets=[
                # {{_base_.train_rain_acdc}},
                # {{_base_.train_fog_acdc}},
                # {{_base_.train_night_acdc}},
                # {{_base_.train_rain_acdc}},
                {{_base_.train_tsp6k}},

            ],
        ),
)


val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_tsp6k}},
            # {{_base_.val_rain_acdc}},
            # {{_base_.val_fog_acdc}},
            # {{_base_.val_night_acdc}},
            # {{_base_.val_rain_acdc}},

        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["tsp", "rain_acdc", "fog_acdc", "night_acdc", "rain_acdc"]
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "tsp", "team"]
# type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["team_seg"]
)
test_evaluator=val_evaluator
