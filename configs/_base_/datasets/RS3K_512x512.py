team_seg_type = "CityscapesDataset"
team_seg_root = "/data/DL/code/atm/data/team_seg/"
team_seg_crop_size = (512, 512)
team_seg_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=team_seg_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
team_seg_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_team_seg = dict(
    type=team_seg_type,
    data_root=team_seg_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=team_seg_train_pipeline,
)
val_team_seg = dict(
    type=team_seg_type,
    data_root=team_seg_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val",
    ),
    img_suffix=".png",
    seg_map_suffix=".png",
    pipeline=team_seg_test_pipeline,
)
