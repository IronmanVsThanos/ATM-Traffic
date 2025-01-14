tsp_type = "CityscapesDataset"
tsp_root = "/data/DL/code/Rein/data/tsp_trainval"
tsp_crop_size = (512, 512)
tsp_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=tsp_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
tsp_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_tsp6k = dict(
    type=tsp_type,
    data_root=tsp_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train_ori",
    ),
    img_suffix=".jpg",
    seg_map_suffix="_sem.png",
    pipeline=tsp_train_pipeline,
)
val_tsp6k = dict(
    type=tsp_type,
    data_root=tsp_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val_ori",
    ),
    img_suffix=".jpg",
    seg_map_suffix="_sem.png",
    pipeline=tsp_test_pipeline,
)
