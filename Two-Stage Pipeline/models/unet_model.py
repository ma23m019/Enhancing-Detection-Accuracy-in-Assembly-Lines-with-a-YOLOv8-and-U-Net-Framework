import segmentation_models_pytorch as smp

def get_unet_model(num_classes=8):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
