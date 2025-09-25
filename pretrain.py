import os
import argparse
import time
import math
import torch
from torch import optim, nn
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import SpongeBob
from Config import LLMConfig
from dataset import PretrainDataset

def get_lr(current_step, total_steps, lr):
    """学习率调度函数"""
    if current_step < total_steps * 0.1:  # 前10%步数使用线性warmup
        return lr * (current_step / (total_steps * 0.1))
    return lr * 0.1 + 0.5 * lr * (1 + math.cos(math.pi * (current_step - total_steps * 0.1) / (total_steps * 0.9)))

def save_checkpoint(model, optimizer, scaler, epoch, step, global_step, loss, config, save_path):
    """保存完整的训练状态"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'config': config.__dict__,
        'timestamp': time.time()
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """加载训练状态"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return 0, 0, 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
    
    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # 加载梯度缩放器状态
    if 'scaler_state_dict' in checkpoint and scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("Scaler state loaded")
    
    # 返回训练状态
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    global_step = checkpoint.get('global_step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Resumed from checkpoint: epoch {epoch}, step {step}, global_step {global_step}, loss {loss:.4f}")
    return epoch, step, global_step, loss

def train_epoch(epoch, start_step, global_step, model, optimizer, scaler, train_loader, args, ctx, wandb):
    """训练一个epoch"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    total_batches = len(train_loader)
     
    # 从指定步骤开始训练
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if step < start_step:
            if step % 100 == 0:  # 每100步打印一次跳过的信息
                print(f"Skipping step {step}/{start_step}")
            continue
               
        X = X.to(args.device,non_blocking=True)
        Y = Y.to(args.device,non_blocking=True)
        loss_mask = loss_mask.to(args.device,non_blocking=True)

        lr = get_lr(global_step, args.total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
          
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
               
        global_step += 1
          
        if step % args.log_step == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            batches_per_sec = (step + 1 - start_step) / spend_time if spend_time > 0 else 0
            eta_minutes = (total_batches - step) / batches_per_sec / 60 if batches_per_sec > 0 else 0
            print(
            'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} global_step:{} ETA:{:.1f}min'.format(
                epoch + 1,
                args.epochs,
                step,
                total_batches,
                current_loss,
                optimizer.param_groups[-1]['lr'],
                global_step,
                eta_minutes
                )
            )
               
        if wandb is not None:
            wandb.log({
                "loss": current_loss,
                "lr": optimizer.param_groups[-1]['lr'],
                "global_step": global_step,
                "epoch": epoch
            })

        # 定期保存checkpoint
        if global_step % args.save_step == 0:
            checkpoint_path = f'{args.save_dir}/checkpoint_epoch_{epoch}_step_{global_step}.pth'
            save_checkpoint(
                model, optimizer, scaler, epoch, step, global_step, 
                current_loss, args.lm_config, checkpoint_path
            )
            
            # 同时保存最新checkpoint
            latest_path = f'{args.save_dir}/latest_checkpoint.pth'
            save_checkpoint(
                model, optimizer, scaler, epoch, step, global_step,
                current_loss, args.lm_config, latest_path
            )

        # 保存最终模型
        if args.save_final_model and (epoch == args.epochs - 1 and step == len(train_loader) - 1):
            model_path = f'{args.save_dir}/pretrain_final.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Final model saved to {model_path}")

    return global_step

def init_model_and_optimizer(args, checkpoint_path=None):
    """初始化模型和优化器，可选择从checkpoint恢复"""
    tokenizer = AutoTokenizer.from_pretrained('./spongebob_tokenizer')
    model = SpongeBob(args.lm_config).to(args.device)
    
    # 初始化优化器和缩放器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda',enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 训练状态变量
    start_epoch = 0
    start_step = 0
    global_step = 0
    best_loss = float('inf')
    
    # 从checkpoint恢复
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, start_step, global_step, best_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scaler, args.device
        )
    
    print(f'LLM parameters size:{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} Million')
    return model, tokenizer, optimizer, scaler, start_epoch, start_step, global_step, best_loss

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--wandb_project", type=str, default="SpongeBob-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--data_path", type=str, default="datasets/pretrain.jsonl")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_final_model", type=bool, default=True, help="Save final model separately")
    
    args = parser.parse_args()

    args.lm_config = LLMConfig(max_seq_len=args.max_seq_len)
    args.save_dir = os.path.join(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"SponseBob-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')

    if args.use_wandb:
        import swanlab as wandb 
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
    else:
        wandb = None
    
    # 初始化模型和优化器（可能从checkpoint恢复）
    model, tokenizer, optimizer, scaler, start_epoch, start_step, global_step, best_loss = init_model_and_optimizer(
    args, args.resume_from
    )

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # 计算总步数（用于学习率调度）
    batches_per_epoch = len(train_loader)
    args.total_steps = args.epochs * batches_per_epoch // args.accumulation_steps
    print(f"Total batches per epoch: {batches_per_epoch}")
    print(f"Total training steps: {args.total_steps}")
    print(f"Starting from epoch {start_epoch}, step {start_step}, global_step {global_step}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Starting Epoch {epoch + 1}/{args.epochs} ===")
    
        global_step = train_epoch(
            epoch, start_step, global_step, model, optimizer, scaler, 
            train_loader, args, ctx, wandb
        )
    
        # 重置start_step，下一epoch从0开始
        start_step = 0
        
        # 每个epoch结束后保存checkpoint
        epoch_checkpoint_path = f'{args.save_dir}/epoch_{epoch+1}_checkpoint.pth'
        save_checkpoint(
            model, optimizer, scaler, epoch + 1, 0, global_step,
            best_loss, args.lm_config, epoch_checkpoint_path
        )
        
        print(f"=== Finished Epoch {epoch + 1}/{args.epochs} ===\n")
    print("Training completed!")


if __name__ == '__main__':
    main()