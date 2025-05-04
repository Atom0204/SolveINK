import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_structure(image_dir, annotation_dir, output_dir, train_csv, val_csv):
    # Ensure directories exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Collect image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    annotations = []
    missing_annotations = []
    invalid_annotations = []

    # Process each image
    for img in image_files:
        annotation_file = os.path.join(annotation_dir, img.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(annotation_file):
            missing_annotations.append(img)
            continue
        
        with open(annotation_file, 'r') as f:
            transcription = f.read().strip()
        
        if not transcription:
            invalid_annotations.append(img)
            continue
        
        annotations.append({
            'image_path': os.path.join('images', img),
            'transcription': transcription
        })

    if not annotations:
        raise ValueError("No valid annotations found. Check annotation files.")

    if missing_annotations:
        print(f"Warning: {len(missing_annotations)} images missing annotations: {missing_annotations[:5]}...")
    if invalid_annotations:
        print(f"Warning: {len(invalid_annotations)} empty annotations: {invalid_annotations[:5]}...")

    # Create DataFrame
    df = pd.DataFrame(annotations)

    # Split into train and validation sets (80% train, 20% val)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save CSV files
    train_path = os.path.join(output_dir, train_csv)
    val_path = os.path.join(output_dir, val_csv)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Dataset created: {len(train_df)} training samples in {train_path}")
    print(f"Dataset created: {len(val_df)} validation samples in {val_path}")

if __name__ == '__main__':
    try:
        create_dataset_structure(
            image_dir='math_ocr_dataset/images',
            annotation_dir='math_ocr_dataset/annotations',
            output_dir='math_ocr_dataset',
            train_csv='train.csv',
            val_csv='val.csv'
        )
    except Exception as e:
        print(f"Error: {str(e)}")