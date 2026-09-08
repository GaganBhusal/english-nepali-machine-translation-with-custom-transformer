import torch

def beam_search_decode(model, source, device, vocab_size_target, max_len_train, sp_english, sp_nepali, beam_size=4, val=False):
    if val:
        source = source[:1].to(device)

    src_enc, src_mask = model.encoder(source)
    
    beam_probs = torch.full((beam_size, 1), -1e9).to(device)
    beam_probs[0] = 0.0
    beam_seq = torch.full((beam_size, 1), 2, dtype=torch.long).to(device)

    with torch.no_grad():
        src_enc_repeat = src_enc.repeat_interleave(beam_size, dim=0)
        src_mask_repeat = src_mask.repeat_interleave(beam_size, dim=0)

        for step in range(max_len_train):
            output = model.decoder(beam_seq, src_enc_repeat, src_mask_repeat)
            output = output[:, -1, :]

            prob_of_last_token = torch.log_softmax(output, dim=-1)
            is_finished = beam_seq[:, -1] == 3

            if is_finished.all():
                break
            
            if is_finished.any():
                prob_of_last_token[is_finished, :] = -float('inf')
                prob_of_last_token[is_finished, 3] = 0.0
            
            current_token_probs = beam_probs + prob_of_last_token
            top_probs, top_tokens = torch.topk(current_token_probs.view(-1), k=beam_size, dim=-1)

            beam_indices = top_tokens // vocab_size_target
            token_ids = top_tokens % vocab_size_target

            beam_indices_expanded = beam_indices.unsqueeze(1).expand(-1, beam_seq.size(1))
            beam_seq = torch.gather(beam_seq, 0, beam_indices_expanded)
            beam_seq = torch.cat([beam_seq, token_ids.unsqueeze(1)], dim=1)
            beam_probs = top_probs.unsqueeze(1)

    best_idx = torch.argmax(beam_probs)
    best_seq = beam_seq[best_idx]
    token_list = best_seq.tolist()
    clean_tokens = [tok for tok in token_list if tok not in [1, 2, 3]]
    
    final_translation = sp_nepali.decode(clean_tokens)



    if val:
        source_tokens = [tok for tok in source[0].tolist() if tok not in [0, 1, 2, 3]]
        source_words = sp_english.decode(source_tokens)
        print(f"ORIGINAL ENGLISH: {source_words}")
        print(f"PREDICTED NEPALI: {final_translation}")
        print("-" * 80)
    else:
        return final_translation
