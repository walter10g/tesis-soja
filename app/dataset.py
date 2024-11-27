import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_paths = []
        self.mask_paths = []

        for category in os.listdir(image_dir):
            category_image_dir = os.path.join(image_dir, category)
            category_mask_dir = os.path.join(mask_dir, category)

            if not os.path.exists(category_mask_dir):
                raise ValueError(f"No se encontró la carpeta de máscaras para {category_image_dir}")

            image_files = sorted(
                [f for f in os.listdir(category_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            )
            mask_files = sorted(
                [f for f in os.listdir(category_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            )

            if len(image_files) != len(mask_files):
                raise ValueError(
                    f"El número de imágenes ({len(image_files)}) no coincide con el número de máscaras "
                    f"({len(mask_files)}) en la categoría {category}"
                )

            self.image_paths.extend([os.path.join(category_image_dir, f) for f in image_files])
            self.mask_paths.extend([os.path.join(category_mask_dir, f) for f in mask_files])

        if len(self.image_paths) == 0:
            raise ValueError(f"No se encontraron imágenes en la carpeta: {image_dir}")
        if len(self.mask_paths) == 0:
            raise ValueError(f"No se encontraron máscaras en la carpeta: {mask_dir}")

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No se encontró la máscara para {img_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Escala de grises

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Asegúrate de que las máscaras contengan índices válidos
        mask = mask.squeeze(0).long()  # Quitar dimensión adicional
        mask = mask % 5  # Asegurar que las clases estén entre 0 y 4 (si tienes 5 clases)

        return image, mask
