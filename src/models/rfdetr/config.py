RFDETR_TRAIN_ARGS = {
    # =========================
    # Core training
    # =========================
    "epochs": 450,
    "batch_size": 32,                        # YOLO batch=32; RF-DETR daha çok VRAM ister, güvenli başlangıç
    "grad_accum_steps": 4,                  # effective batch = 8 * 4 = 32 → YOLO batch=32'ye denk
    "auto_batch_target_effective": 32,      # batch_size="auto" kullanırsan hedef effective batch

    "resolution": 1280,                     # Bu model varyantında çözünürlük 32'ye tam bölünmeli.
                                            # Smoke/local eğitim için YOLO imgsz=1280 ile hizalı tutuyoruz.

    "dataset_file": "roboflow",             # COCO klasör yapısı için genelde roboflow/coco kullanılır.
                                            # Datasetin direkt COCO ise "coco" da denenebilir.

    "output_dir": None,                     # fold bazında dinamik verilecek
    "resume": None,                         # YOLO resume=False karşılığı

    # =========================
    # Device / performance
    # =========================
    "accelerator": "auto",                  # YOLO device=0 birebir değil; Lightning auto GPU seçer
    "devices": 1,                           # YOLO device=0'a en yakın: tek GPU
    "num_workers": 8,
    "pin_memory": True,                     # DataLoader hızlandırma
    "persistent_workers": True,             # workers tekrar tekrar kapanıp açılmasın
    "prefetch_factor": 2,                   # DataLoader prefetch

    # =========================
    # Optimizer / scheduler
    # =========================
    "lr": 4e-4,                             # YOLO lr0=0.004 doğrudan taşınmaz.
                                            # Transformer fine-tuning için daha düşük LR tercih edilir.

    "lr_encoder": 1.5e-4,                   # Encoder/backbone için ayrı LR.
                                            # YOLO'da birebir karşılığı yok.

    "weight_decay": 0.0005,                 # YOLO weight_decay=0.0005 ile aynı

    "lr_scheduler": "step",                 # YOLO cos_lr=False karşılığı
    "lr_drop": 350,                         # LR düşürme epoch'u. 450 epoch için geç düşüş.
    "lr_min_factor": 0.01,                  # YOLO lrf=0.01'e en yakın mantık
    "warmup_epochs": 10.0,                  # YOLO warmup_epochs=10.0 karşılığı

    # =========================
    # Regularization / stability
    # =========================
    "clip_max_norm": 0.1,                   # Transformer training stabilitesi için gradient clipping
    "drop_path": 0.0,                       # YOLO dropout=0.0'a en yakın regularization karşılığı
    "seed": 0,                              # YOLO seed=0 karşılığı
    "sync_bn": False,                       # Tek GPU için gerek yok

    # =========================
    # EMA / checkpoints
    # =========================
    "use_ema": True,                        # YOLO'da doğrudan arg yok ama genelde faydalı
    "ema_decay": 0.993,
    "ema_tau": 100,
    "ema_update_interval": 1,

    "checkpoint_interval": 10,              # YOLO save_period=-1 birebir değil.
                                            # RF-DETR checkpoint_interval >= 1 ister.

    "dont_save_weights": False,             # YOLO save=True karşılığı

    # =========================
    # Validation / evaluation
    # =========================
    "eval_interval": 1,                     # YOLO val=True gibi her epoch validation
    "eval_max_dets": 300,                   # YOLO max_det=300 karşılığı
    "run_test": False,                      # Eğitim sonunda test set çalıştırmak istersen True
    "compute_val_loss": True,
    "compute_test_loss": True,
    "log_per_class_metrics": True,

    # !!! YOLO iou=0.7:
    # RF-DETR train config içinde doğrudan validation IoU threshold arg'ı yok.
    # COCO mAP evaluation farklı IoU aralıklarıyla yapılır.

    # ? YOLO half=False / amp=True:
    # RF-DETR high-level config'te train AMP flag'i birebir yok.
    # fp16_eval sadece evaluation tarafı için var.
    "fp16_eval": False,

    # =========================
    # Early stopping
    # =========================
    "early_stopping": True,                 # YOLO patience karşılığı
    "early_stopping_patience": 200,         # YOLO patience=200 karşılığı
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,

    # =========================
    # Logging / visualization
    # =========================
    "progress_bar": "tqdm",                 # YOLO verbose=True karşılığı.
                                            # Terminalde training progress, epoch ilerlemesi ve metrikleri gösterir.

    "tensorboard": True,                    # YOLO plots=True'ye en yakın karşılık.
                                            # RF-DETR direkt YOLO gibi results.png üretmez.
                                            # Bunun yerine TensorBoard logları üretir.

    "wandb": False,                         # İstersen ileride True yapıp cloud experiment tracking açabiliriz.

    "mlflow": False,                        # Lokal/kurumsal experiment tracking için alternatif.

    "project": "lens-damage-rfdetr",
    "run": None,                            # Fold içinde dinamik olarak fold_1, fold_2... verilecek.

    # =========================
    # Multi-scale / resize
    # =========================
    "multi_scale": False,                   # YOLO multi_scale=False karşılığı
    "expanded_scales": False,               # multi_scale kapalıyken agresif scale aralığı da kapalı
    "square_resize_div_64": True,
    "do_random_resize_via_padding": False,

    # =========================
    # Augmentation
    # =========================
    "aug_config": {
        # YOLO fliplr=0.5
        "HorizontalFlip": {"p": 0.5},

        # YOLO flipud=0.5
        "VerticalFlip": {"p": 0.5},

        # YOLO translate=0.05, scale=0.25 için yaklaşık karşılık.
        # Albumentations Affine kullanılır.
        "Affine": {
            "translate_percent": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "scale": (0.75, 1.25),
            "rotate": 0,
            "shear": 0,
            "p": 0.5,
        },

        # YOLO hsv_h=0.01, hsv_s=0.4, hsv_v=0.25 yaklaşık karşılığı
        "HueSaturationValue": {
            "hue_shift_limit": 4,
            "sat_shift_limit": 40,
            "val_shift_limit": 25,
            "p": 0.5,
        },
    },

    # ? YOLO box, cls, dfl:
    # RF-DETR'de cls_loss_coef var ama box/dfl birebir YOLO gibi değil.
    # "cls_loss_coef": 1.0,

    # YOLO max_det=300 karşılığı yukarıda eval_max_dets=300.
    # RF-DETR num_select detection seçimi:
    "num_select": 300,
}
