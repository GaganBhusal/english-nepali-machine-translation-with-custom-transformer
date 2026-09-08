# train script
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from src import MyTransformer, TransformerLRScheduler
from src.data.dataset_process import DatasetProcess
from scripts.beam_search import beam_search_decode
import argparse
import os

def train(epochs, train_loader, val_loader, max_len_train, sp_english, sp_nepali, d_model, heads, dff, dropout, beam_size = 4):

    seq_length_source = max_len_train
    seq_length_target = max_len_train

    vocab_size_source = sp_english.vocab_size()
    vocab_size_target = sp_nepali.vocab_size()

    d_model = d_model
    heads = heads
    dropout = dropout
    dff = dff

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = MyTransformer(
        vocab_size_source,
        vocab_size_target,
        seq_length_source,
        seq_length_target,
        device,
        d_model,
        heads,
        dropout,
        dff
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=4000)
    criteria = nn.CrossEntropyLoss(ignore_index=0)


    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            
    model.apply(init_weights)

    def validation(model, val_loader, epoch, device, vocab_size_target, max_len_train, sp_english, sp_nepali, beam_size):
        model.eval()
        val_loss = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False)

        with torch.no_grad():
            for i, (source, target_ip, target_op) in enumerate(val_bar):
                source = source.to(device)
                target_ip = target_ip.to(device)
                target_op = target_op.to(device)

                output = model(source, target_ip)
                output = output.reshape(-1, output.size(-1))
                target_op = target_op.reshape(-1)

                loss = criteria(output, target_op)
                val_loss += loss.item()
                
                val_bar.set_postfix(loss=f"{val_loss / (i+1):.4f}")
                
                if i < 4:
                    print(f"Beam Search Sample {i+1}:")
                    beam_search_decode(model, source, device, vocab_size_target, max_len_train, sp_english, sp_nepali, val=True, beam_size=beam_size)

        return val_loss / len(val_loader)


    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        
        for i, (source, target_ip, target_op) in enumerate(train_bar):
            source = source.to(device)
            target_ip = target_ip.to(device)
            target_op = target_op.to(device)

            optimizer.zero_grad()
            output = model(source, target_ip)
            
            output = output.reshape(-1, output.size(-1))
            target_op = target_op.reshape(-1)

            loss = criteria(output, target_op)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            train_bar.set_postfix(loss=f"{train_loss / (i+1):.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validation(model,
                        val_loader, 
                        epoch, 
                        device, 
                        vocab_size_target, 
                        max_len_train, 
                        sp_english, 
                        sp_nepali,
                        beam_size
                    )
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_step_num': scheduler.step_num,
            'history': history
        }
        torch.save(checkpoint, os.path.join("checkpoints", "transformer_nepali_checkpoint1.pth"))
        
    return history, model, optimizer, scheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from Scratch")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16,)
    parser.add_argument("--max_len", type=int, default=64,)
    parser.add_argument("--vocab_size", type=int, default=16000)
    
    parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dff", type=int, default=2048, help="FeedForward hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size for validation")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    dataset = DatasetProcess(max_len=args.max_len, vocab_size=args.vocab_size)
    dataset.process_dataloader(batch_size=args.batch_size)
    
    train(
        epochs=args.epochs,
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        max_len_train=args.max_len,
        sp_english=dataset.sp_english,
        sp_nepali=dataset.sp_nepali,
        d_model=args.d_model,
        heads=args.heads,
        dff=args.dff,
        dropout=args.dropout,
        beam_size=args.beam_size
    )