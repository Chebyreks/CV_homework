import cv2
import numpy as np
import random
import os
import shutil
from pathlib import Path
import json
import albumentations as A
from typing import Tuple, List, Dict, Optional
import imutils

class SyntheticDatasetGenerator:
    def __init__(self, 
                 base_path: str = "data",
                 output_path: str = "dataset",
                 image_size: Tuple[int, int] = (256, 256),
                 seed: int = 42):

        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.seed = seed
        self.backgrounds = self._load_backgrounds()
        self.animals = self._load_animals()

        self.class_config = {
            'dog': {
                'preferred_backgrounds': ['lawn', 'house', 'garbage'],
                'scale_range': (0.45, 0.45),
                'brightness_range': (-20, 20),
            },
            'cat': {
                'preferred_backgrounds': ['house', 'lawn'],
                'scale_range': (0.4, 0.4),
                'brightness_range': (-15, 15),
            },
            'capybara': {
                'preferred_backgrounds': ['pond', 'lawn'],
                'scale_range': (0.5, 0.6),
                'brightness_range': (-10, 20),
            }
        }
        
        self.bg_augmentations = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.GaussNoise(var_limit=(1, 10), p=0.1)
        ], p=0.7)

    def _load_backgrounds(self) -> Dict[str, List[np.ndarray]]:
        backgrounds = {}
        bg_path = self.base_path / "backgrounds"
        
        for category_dir in bg_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                backgrounds[category] = []

                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
                
                for ext in extensions:
                    for bg_file in category_dir.glob(ext):
                        try:
                            bg = cv2.imread(str(bg_file))
                            if bg is not None:
                                bg = cv2.resize(bg, self.image_size)
                                backgrounds[category].append(bg)
                        except Exception as e:
                            print(f"Ошибка загрузки фона {bg_file}: {e}")
                
                print(f"  {category}: {len(backgrounds[category])} фонов")
        
        return backgrounds
    
    def _load_animals(self) -> Dict[str, List[np.ndarray]]:
        animals = {}
        classes_path = self.base_path / "classes"
        
        for animal_dir in classes_path.iterdir():
            if animal_dir.is_dir():
                animal = animal_dir.name
                animals[animal] = []

                for animal_file in animal_dir.glob('*.png'):
                    try:
                        img = cv2.imread(str(animal_file), cv2.IMREAD_UNCHANGED)
                        animals[animal].append(img)     
                    except Exception as e:
                        print(f"Ошибка загрузки животного {animal_file}: {e}")
        return animals
    
    def _augment_animal(self, 
                       animal_img: np.ndarray, 
                       class_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if animal_img.shape[2] == 4:
            rgb = animal_img[:, :, :3]
            alpha = animal_img[:, :, 3]
        else:
            rgb = animal_img
            alpha = np.ones((animal_img.shape[0], animal_img.shape[1]), dtype=np.uint8) * 255
        
        config = self.class_config.get(class_name, {})
        
        scale_min, scale_max = config.get('scale_range', (0.3, 0.5))
        scale = random.uniform(scale_min, scale_max)
        height, width = rgb.shape[:2]
        new_h, new_w = int(height * scale), int(width * scale)
        
        if new_h > 0 and new_w > 0:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        max_rot = 30
        angle = random.uniform(-max_rot, max_rot)
        
        if abs(angle) > 1:
            rgb = imutils.rotate_bound(rgb, angle)
            alpha = imutils.rotate_bound(alpha, angle)
        
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            alpha = cv2.flip(alpha, 1)
        
        if random.random() > 0.3:
            brightness_range = config.get('brightness_range', (-20, 20))
            brightness = random.randint(brightness_range[0], brightness_range[1])
            contrast = random.uniform(0.8, 1.2)
            rgb = cv2.convertScaleAbs(rgb, alpha=contrast, beta=brightness)
        
        if random.random() > 0.9:
            kernel_size = 3
            rgb = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
        
        return rgb, alpha
    
    def _select_background(self, class_name: str) -> Tuple[np.ndarray, str]:
        config = self.class_config.get(class_name, {})
        preferred = config.get('preferred_backgrounds', list(self.backgrounds.keys()))

        if random.random() < 0.7 and preferred:
            available_preferred = [cat for cat in preferred 
                                 if cat in self.backgrounds and len(self.backgrounds[cat]) > 0]
            
            if available_preferred:
                category = random.choice(available_preferred)
                bg = random.choice(self.backgrounds[category]).copy()
                return bg, category
        
        all_backgrounds = []
        all_categories = []
        
        for category, bg_list in self.backgrounds.items():
            if bg_list:
                all_backgrounds.extend(bg_list)
                all_categories.extend([category] * len(bg_list))

        idx = random.randint(0, len(all_backgrounds) - 1)
        return all_backgrounds[idx].copy(), all_categories[idx]
    
    def _paste_animal_on_background(self,
                                  animal_rgb: np.ndarray,
                                  animal_alpha: np.ndarray,
                                  background: np.ndarray) -> np.ndarray:
        bg_h, bg_w = background.shape[:2]
        animal_h, animal_w = animal_rgb.shape[:2]
        
        max_size = min(bg_h, bg_w) * 0.7
        if animal_h > max_size or animal_w > max_size:
            scale = max_size / max(animal_h, animal_w)
            new_h, new_w = int(animal_h * scale), int(animal_w * scale)
            animal_rgb = cv2.resize(animal_rgb, (new_w, new_h))
            animal_alpha = cv2.resize(animal_alpha, (new_w, new_h))
            animal_h, animal_w = new_h, new_w

        margin = 20
        max_x = max(1, bg_w - animal_w - margin)
        max_y = max(1, bg_h - animal_h - margin)
        
        if max_x <= 0 or max_y <= 0:
            x = max(0, (bg_w - animal_w) // 2)
            y = max(0, (bg_h - animal_h) // 2)
        else:
            x = random.randint(margin, max_x)
            y = random.randint(margin, max_y)

        alpha_normalized = animal_alpha.astype(float) / 255.0

        if len(alpha_normalized.shape) == 2:
            alpha_normalized = alpha_normalized[:, :, np.newaxis]

        roi = background[y:y+animal_h, x:x+animal_w]

        if roi.shape[:2] == animal_rgb.shape[:2]:
            for c in range(3):
                roi[:, :, c] = (roi[:, :, c] * (1 - alpha_normalized[:, :, 0]) + 
                              animal_rgb[:, :, c] * alpha_normalized[:, :, 0])
            
            background[y:y+animal_h, x:x+animal_w] = roi
        
        return background
    
    def generate_synthetic_image(self, class_name: str) -> np.ndarray:
        animal_idx = random.randint(0, len(self.animals[class_name]) - 1)
        animal_img = self.animals[class_name][animal_idx].copy()

        animal_rgb, animal_alpha = self._augment_animal(animal_img, class_name)
  
        background, bg_category = self._select_background(class_name)
        
        if random.random() < 0.7:
            augmented = self.bg_augmentations(image=background)
            background = augmented['image']
        
        result = self._paste_animal_on_background(animal_rgb, animal_alpha, background)
        
        if result.shape[:2] != self.image_size:
            result = cv2.resize(result, self.image_size)
        
        return result
    
    def create_dataset_splits(self,
                            train_per_class: int = 700,
                            val_per_class: int = 150,
                            test_per_class: int = 150,
                            clear_existing: bool = True):
        if clear_existing and self.output_path.exists():
            shutil.rmtree(self.output_path)

        splits = ['train', 'validation', 'test']
        for split in splits:
            split_dir = self.output_path / split
            split_dir.mkdir(exist_ok=True, parents=True)
            
            for class_name in self.animals.keys():
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True, parents=True)

        split_config = {
            'train': train_per_class,
            'validation': val_per_class,
            'test': test_per_class
        }
        
        statistics = {
            'total_generated': 0,
            'per_class': {},
            'per_split': {}
        }

        for class_name in self.animals.keys():
            class_stats = {}
            
            for split_name, num_images in split_config.items():
                split_dir = self.output_path / split_name / class_name
                generated_count = 0
                attempts = 0
                max_attempts = num_images * 2
                while generated_count < num_images and attempts < max_attempts:
                    try:
                        synthetic_img = self.generate_synthetic_image(class_name)
                        
                        img_filename = f"{class_name}_{split_name}_{generated_count:04d}.jpg"
                        img_path = split_dir / img_filename
                        
                        cv2.imwrite(str(img_path), synthetic_img, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        generated_count += 1
                        
                    except Exception as e:
                        attempts += 1
                        if attempts % 50 == 0:
                            print(f"\n    Предупреждение: ошибка при генерации ({e})")
                    
                    attempts += 1
                
                class_stats[split_name] = generated_count
                if generated_count < num_images:
                    print(f"Сгенерировано только {generated_count} из {num_images}!")
            
            statistics['per_class'][class_name] = class_stats
        
        total_images = 0
        for split_name in splits:
            split_total = 0
            for class_stats in statistics['per_class'].values():
                split_total += class_stats.get(split_name, 0)
            statistics['per_split'][split_name] = split_total
            total_images += split_total
        
        statistics['total_generated'] = total_images
        self._print_final_statistics(statistics)
    
    def _print_final_statistics(self, statistics: Dict):
        print(f"\nОбщее количество изображений: {statistics['total_generated']}")
        
        print("\nПо классам:")
        for class_name, class_stats in statistics['per_class'].items():
            class_total = sum(class_stats.values())
            print(f"  {class_name:10s}: {class_total:4d} изображений")
            for split_name, count in class_stats.items():
                print(f"    {split_name:12s}: {count:4d}")
        
        print("\nПо разделам:")
        for split_name, count in statistics['per_split'].items():
            print(f"  {split_name:12s}: {count:4d} изображений")

def main():
    CONFIG = {
        'base_path': 'data',
        'output_path': 'synthetic_dataset',
        'image_size': (224, 224),
        'seed': 42,
        'train_per_class': 600,
        'val_per_class': 100,
        'test_per_class': 100
    }
    
    print(f"  Исходные данные: {CONFIG['base_path']}")
    print(f"  Выходной датасет: {CONFIG['output_path']}")
    print(f"  Размер изображений: {CONFIG['image_size']}")
    print(f"  Классы: capybara, cat, dog")
    print(f"  Фоны: garbage, house, lawn, pond")

    generator = SyntheticDatasetGenerator(
        base_path=CONFIG['base_path'],
        output_path=CONFIG['output_path'],
        image_size=CONFIG['image_size'],
        seed=CONFIG['seed']
    )

    generator.create_dataset_splits(
        train_per_class=CONFIG['train_per_class'],
        val_per_class=CONFIG['val_per_class'],
        test_per_class=CONFIG['test_per_class'],
        clear_existing=True
    )

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()