import os
import json
import shutil
import logging
from pathlib import Path

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_coco_to_yolo(dataset_dir):
    """
    Converte un dataset in formato COCO in formato YOLO.
    
    Args:
        dataset_dir: Il percorso alla directory principale del dataset
        
    Returns:
        str: Percorso al file YAML generato per il dataset YOLO
    """
    logger.info(f"Convertendo dataset COCO in YOLO: {dataset_dir}")
    
    # Verifica che la directory del dataset esista
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Verifica quali split sono presenti (train, val, test)
    splits = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            splits.append(split)
    
    if not splits:
        raise ValueError(f"No valid splits (train, val, test) found in {dataset_dir}")
    
    logger.info(f"Found splits: {splits}")
    
    # Crea la struttura delle directory YOLO
    yolo_dataset_dir = os.path.join(os.path.dirname(dataset_dir), f"{os.path.basename(dataset_dir)}_yolo")
    os.makedirs(yolo_dataset_dir, exist_ok=True)
    
    # Dizionario per memorizzare tutte le classi
    all_categories = {}
    
    # Processa ogni split
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        coco_file = os.path.join(split_dir, "_annotations.coco.json")
        
        if not os.path.exists(coco_file):
            logger.warning(f"COCO annotations file not found for {split}: {coco_file}")
            continue
        
        # Carica il file di annotazioni COCO
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Estrai le categorie (classi)
        categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        all_categories.update(categories)
        
        # Crea le directory per le immagini e le etichette YOLO
        yolo_split_dir = os.path.join(yolo_dataset_dir, split)
        yolo_images_dir = os.path.join(yolo_split_dir, "images")
        yolo_labels_dir = os.path.join(yolo_split_dir, "labels")
        
        os.makedirs(yolo_images_dir, exist_ok=True)
        os.makedirs(yolo_labels_dir, exist_ok=True)
        
        # Mappa immagini per ID
        images = {img['id']: img for img in coco_data.get('images', [])}
        
        # Raggruppa le annotazioni per immagine
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Processa ogni immagine
        for img_id, img_info in images.items():
            # Copia l'immagine nella directory YOLO
            img_filename = img_info['file_name']
            src_img_path = os.path.join(split_dir, img_filename)
            dst_img_path = os.path.join(yolo_images_dir, img_filename)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                logger.warning(f"Image not found: {src_img_path}")
                continue
            
            # Crea il file di etichetta YOLO corrispondente
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Nome del file di etichetta (stesso nome del file immagine ma con estensione .txt)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(yolo_labels_dir, label_filename)
            
            # Scrivi le annotazioni in formato YOLO
            with open(label_path, 'w') as f:
                if img_id in annotations_by_image:
                    for ann in annotations_by_image[img_id]:
                        # YOLO format: class_id x_center y_center width height
                        # Values are normalized to [0, 1]
                        cat_id = ann['category_id']
                        # In YOLO, gli ID delle classi iniziano da 0
                        yolo_cat_id = cat_id - 1  # Assumendo che gli ID COCO inizino da 1
                        
                        # Ottieni le coordinate del bounding box
                        bbox = ann['bbox']  # [x, y, width, height] in formato COCO (top-left x, y, width, height)
                        
                        # Converti in formato YOLO (x_center, y_center, width, height) normalizzato
                        x_center = (bbox[0] + bbox[2] / 2) / img_width
                        y_center = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height
                        
                        # Scrivi la riga di annotazione
                        f.write(f"{yolo_cat_id} {x_center} {y_center} {width} {height}\n")
    
    # Crea il file YAML per la configurazione del dataset
    yaml_path = os.path.join(yolo_dataset_dir, "data.yaml")
    
    # Prepara la configurazione YAML
    yaml_config = {
        "path": os.path.abspath(yolo_dataset_dir),
        "train": "train/images" if "train" in splits else "",
        "val": "val/images" if "val" in splits else "",
        "test": "test/images" if "test" in splits else "",
        "names": {}
    }
    
    # Aggiungi i nomi delle classi
    for cat_id, cat_info in sorted(all_categories.items(), key=lambda x: x[0]):
        # In YOLO, gli ID delle classi iniziano da 0
        yolo_cat_id = cat_id - 1  # Assumendo che gli ID COCO inizino da 1
        yaml_config["names"][yolo_cat_id] = cat_info.get('name', f"class_{cat_id}")
    
    # Scrivi il file YAML
    with open(yaml_path, 'w') as f:
        yaml_string = ""
        yaml_string += f"path: {yaml_config['path']}\n"
        yaml_string += f"train: {yaml_config['train']}\n"
        yaml_string += f"val: {yaml_config['val']}\n"
        yaml_string += f"test: {yaml_config['test']}\n"
        
        # Scrivi i nomi delle classi
        yaml_string += "names:\n"
        for class_id, class_name in yaml_config["names"].items():
            yaml_string += f"  {class_id}: {class_name}\n"
        
        f.write(yaml_string)
    
    logger.info(f"Dataset YOLO creato con successo: {yolo_dataset_dir}")
    logger.info(f"File di configurazione YAML: {yaml_path}")
    logger.info(f"Numero di classi: {len(yaml_config['names'])}")
    
    return yaml_path


# Modifiche da apportare a yolo_training.py:
# 
# 1. Aggiungere l'importazione:
#    from coco_to_yolo_converter import convert_coco_to_yolo
#
# 2. Aggiungere nel metodo train_yolo_model, prima della configurazione del dataset:
#
#    # Controlla se il dataset è in formato COCO
#    is_coco_format = False
#    for split in ['train', 'val', 'test']:
#        coco_file = os.path.join(dataset_path, split, "_annotations.coco.json")
#        if os.path.exists(coco_file):
#            is_coco_format = True
#            break
#    
#    # Se il dataset è in formato COCO, convertilo in YOLO
#    if is_coco_format:
#        logger.info(f"Detected COCO format dataset at {dataset_path}. Converting to YOLO format...")
#        try:
#            dataset_path = convert_coco_to_yolo(dataset_path)
#            logger.info(f"Dataset successfully converted to YOLO format. Using: {dataset_path}")
#        except Exception as e:
#            logger.error(f"Failed to convert COCO dataset to YOLO format: {str(e)}")
#            logger.error(f"Detailed traceback: {traceback.format_exc()}")
#            # Continua con il dataset originale
#            logger.warning(f"Continuing with original dataset path: {dataset_path}")

# Esempio di utilizzo:
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
        convert_coco_to_yolo(dataset_dir)
    else:
        print("Utilizzo: python coco_to_yolo_converter.py <percorso_dataset>")
