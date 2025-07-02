import os
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import wandb

from mvadapt_model import MVAdapt
from mvadapt_data import MVAdaptDataset

def compute_wp_accuracy(predicted, gt, tolerance=0.05):
    dimension_accuracy = torch.abs(predicted - gt) < tolerance
    correct = dimension_accuracy.all(dim=1)
    return correct.sum().item() / gt.size(0)

def lossfn(pred, gt):
    return torch.mean(torch.abs(pred - gt))

def validate(model, args, dataset) -> None:
    print("Validating Model")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Validation"):
            scene_feature = data['scene_features'].to(args.device, dtype=torch.float32)
            phys = data['physics_params'].to(args.device, dtype=torch.float32)
            gear = data['gear_params'].to(args.device, dtype=torch.float32)
            target = data['target_point'].to(args.device, dtype=torch.float32)
            gt_wp = data['gt_waypoint'].to(args.device, dtype=torch.float32)
            
            predicted = model.inference(scene_feature, target, phys, gear)
            loss = lossfn(predicted, gt_wp)
            total_loss += loss.item()
            
            batch_accuracy = compute_wp_accuracy(predicted, gt_wp, tolerance=0.05)
            total_correct += batch_accuracy * gt_wp.size(0)
            total_samples += gt_wp.size(0)
            
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_samples
    print(f"Validation Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    wandb.log({"validation_loss": avg_loss, "validation_accuracy": avg_accuracy})

def finetune(model, optimizer, args, dataset):
    print("Fine-tuning Model...")
    model.to(args.device)
    loss_fn = lossfn

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.finetune_epochs):
        model.train() 
        
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Fine-tuning Epoch {epoch + 1}")

        for data in progress_bar:
            rgb = data['rgb'].to(args.device, dtype=torch.float32)
            scene_feature = data['scene_features'].to(args.device, dtype=torch.float32)
            phys = data['physics_params'].to(args.device, dtype=torch.float32)
            gear = data['gear_params'].to(args.device, dtype=torch.float32)
            target = data['target_point'].to(args.device, dtype=torch.float32)
            gt_wp = data['gt_waypoint'].to(args.device, dtype=torch.float32)
            
            optimizer.zero_grad()
            predicted = model.forward(scene_feature, target, phys, gear)
            loss = loss_fn(predicted, gt_wp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        wandb.log({"epoch": epoch + 1, "finetune_loss": avg_loss})
        print(f"Fine-tuning Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_model", type=str, default="/path/to/base_model")
    parser.add_argument("--process_batch", type=int, default=32)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--unseen_vehicle_id", type=int, required=True)
    parser.add_argument("--load_data", type=str, default=None)
    parser.add_argument("--save_data", type=str, default=None)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--save_finetuned_model", type=str, default="mvadapt_finetuned.pth")

    args = parser.parse_args()
    args.vehicle_indices = f"[{args.unseen_vehicle_id}]"
    
    wandb.init(project="MVAdapt-Finetuning", config=args.__dict__)

    finetune_set = MVAdaptDataset(args.root_dir)
    
    loaded = False
    if args.load_data is not None and args.load_data != "None":
        try:
            finetune_set.load(f"{args.load_data}.pkl")
            loaded = True
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            loaded = False
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            loaded = False
            
    if not loaded:
        finetune_set.initialize(args, 'train')

    if args.save_data is not None and args.save_data != "None":
        finetune_set.save(args.save_data + ".pkl")
    
    val_set = finetune_set 
    print(f"Fine-tuning set size: {len(finetune_set)}")
    print(f"Validation set size: {len(val_set)}")

    model = MVAdapt(finetune_set.config).to(args.device)
    
    try:
        print(f"Loading pre-trained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.physics_encoder.parameters():
        param.requires_grad = True
        
    for param in model.transformer_encoder.parameters():
        param.requires_grad = True

    print("Trainable parameters have been set for fine-tuning:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    # optimizer = optim.Adam( # type: ignore
    #     filter(lambda p: p.requires_grad, model.parameters()), 
    #     lr=args.finetune_lr
    # )
    optimizer = optim.Adam([
        {'params': model.physics_encoder.parameters(), 'lr': args.finetune_lr},
        {'params': model.transformer_encoder.parameters(), 'lr': args.finetune_lr * 0.1} # 어텐션 레이어는 1/10 수준의 학습률 적용
    ])
    
    print("\n--- Evaluating performance BEFORE fine-tuning ---")
    validate(model, args, val_set)

    finetune(model, optimizer, args, finetune_set)
    
    print("\n--- Evaluating performance AFTER fine-tuning ---")
    validate(model, args, val_set)

    if args.save_finetuned_model:
        model.save(args.save_finetuned_model)
        print(f"Fine-tuned model saved to {args.save_finetuned_model}")
        
    wandb.finish()

if __name__ == "__main__":
    main()