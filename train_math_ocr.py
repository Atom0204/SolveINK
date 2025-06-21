import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from easyocr import Reader
from easyocr.utils import CTCLabelConverter
from custom_dataset import CustomMathDataset
import argparse

def train(model, train_loader, val_loader, num_epochs, device, save_path, converter):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        try:
            for images, labels in train_loader:
                images = images.to(device)
                batch_size = images.size(0)
                
                optimizer.zero_grad()
                
                # Create dummy text input (EasyOCR's recognizer expects this)
                text_for_pred = torch.LongTensor(batch_size, 26).fill_(0).to(device)
                
                # Forward pass with both image and text
                # Note: Removed is_train parameter as it's not accepted
                log_probs = model(images, text_for_pred)
                
                # Model outputs (batch_size, sequence_length, num_classes)
                # CTC needs (sequence_length, batch_size, num_classes)
                log_probs = log_probs.permute(1, 0, 2)
                log_probs = log_probs.log_softmax(2)
                
                input_lengths = torch.full(
                    size=(batch_size,), 
                    fill_value=log_probs.size(0), 
                    dtype=torch.long
                ).to(device)
                
                target_lengths = torch.tensor(
                    [len(label) for label in labels], 
                    dtype=torch.long
                ).to(device)
                
                targets = torch.cat(
                    [converter.encode(label)[0] for label in labels]
                ).to(device)
                
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                if device.type == 'cuda':
                    print(f"Batch processed, GPU Memory Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    batch_size = images.size(0)
                    
                    text_for_pred = torch.LongTensor(batch_size, 26).fill_(0).to(device)
                    log_probs = model(images, text_for_pred)
                    log_probs = log_probs.permute(1, 0, 2)
                    log_probs = log_probs.log_softmax(2)
                    
                    input_lengths = torch.full(
                        (batch_size,), 
                        log_probs.size(0), 
                        dtype=torch.long
                    ).to(device)
                    
                    target_lengths = torch.tensor(
                        [len(label) for label in labels], 
                        dtype=torch.long
                    ).to(device)
                    
                    targets = torch.cat(
                        [converter.encode(label)[0] for label in labels]
                    ).to(device)
                    
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()
        
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            
            checkpoint_path = os.path.join(save_path, f'math_ocr_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss/len(train_loader),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        except RuntimeError as e:
            print(f"Error during epoch {epoch+1}: {e}")
            if "out of memory" in str(e).lower():
                print("CUDA out of memory. Try reducing batch size.")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='C:/Users/shiva/SolveINK/EasyOCR/math_ocr_dataset/train.csv')
    parser.add_argument('--val_csv', default='C:/Users/shiva/SolveINK/EasyOCR/math_ocr_dataset/val.csv')
    parser.add_argument('--root_dir', default='C:/Users/shiva/SolveINK/EasyOCR/math_ocr_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', default='model')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--model_dir', default='C:/Users/shiva/SolveINK/EasyOCR/model', help='Directory for model storage')
    parser.add_argument('--label_column', default='transcription', help='Column name for labels in CSV')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Initialize EasyOCR reader
        reader = Reader(['en'], gpu=(device.type == 'cuda'), model_storage_directory=args.model_dir)
        model = reader.recognizer.module if isinstance(reader.recognizer, nn.DataParallel) else reader.recognizer
        model = model.to(device)
        converter = CTCLabelConverter(reader.character)
        print(f"Model loaded with {len(converter.character)} characters")
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        return

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resumed from checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 128)),  # Adjust based on model's expected input size
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        train_dataset = CustomMathDataset(args.train_csv, args.root_dir, transform=transform, label_column=args.label_column)
        val_dataset = CustomMathDataset(args.val_csv, args.root_dir, transform=transform, label_column=args.label_column)
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train(model, train_loader, val_loader, args.epochs, device, args.save_path, converter)

if __name__ == '__main__':
    main()