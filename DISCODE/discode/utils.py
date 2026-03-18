import gc
import torch
from torch.utils.data import DataLoader
import esm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from tqdm.auto import tqdm

_ESM_MODEL = None
_ESM_ALPHABET = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_esm_model():
    global _ESM_MODEL, _ESM_ALPHABET, _DEVICE
    if _ESM_MODEL is None:
        print("Loading ESM model...")
        _ESM_MODEL, _ESM_ALPHABET = esm.pretrained.esm2_t12_35M_UR50D()
        _ESM_MODEL.to(_DEVICE)
        _ESM_MODEL.eval()
    return _ESM_MODEL, _ESM_ALPHABET


def listup_outlier_residues(attention_matrix, threshold="2S"):
    if len(attention_matrix.shape) == 5: 
        att_sum = np.sum(np.sum(np.sum(attention_matrix, axis=1), axis=1), axis=1)
        results = []
        for i in range(att_sum.shape[0]):
            threshold_value = specify_threshold(att_sum[i], threshold)
            idx = np.where(att_sum[i] > threshold_value)[0]
            results.append(idx)
        return results
    else:
        att_sum = np.sum(np.sum(np.sum(attention_matrix, axis=0), axis=0), axis=0)
        threshold_value = specify_threshold(att_sum, threshold)
        idx = np.where((att_sum > threshold_value) == True)[0]
        return idx

def specify_threshold(att_sum, threshold):
    if threshold == "1S":
        threshold_value = np.mean(att_sum) + 1 * np.std(att_sum)
    elif threshold == "2S":
        threshold_value = np.mean(att_sum) + 2 * np.std(att_sum)
    elif threshold == "3S":
        threshold_value = np.mean(att_sum) + 3 * np.std(att_sum)
    elif threshold == "IQR":
        Q1 = np.percentile(att_sum, 25)
        Q3 = np.percentile(att_sum, 75)
        IQR = Q3 - Q1
        threshold_value = Q3 + 1.5 * IQR
    elif threshold == "P90":
        threshold_value = np.percentile(att_sum, 90)
    elif threshold == "P95":
        threshold_value = np.percentile(att_sum, 95)
    elif threshold == "P99":
        threshold_value = np.percentile(att_sum, 99)
    else:
        raise ValueError("Error: Invalid threshold.")
    return threshold_value

def collect_attention_weights(inputs, model):
    # Single input wrapper
    x = inputs
    attention_weights = []
    for layer in model.transformer_encoder.layers:
        _, weight = layer.self_attn(x, x, x, average_attn_weights=False)
        attention_weights.append(weight.squeeze(0).cpu().numpy())
        x = layer(x)
    attention_weights = np.asarray(attention_weights)
    return attention_weights

def collect_attention_weights_batch(inputs, model):
    # Batch input wrapper
    x = inputs
    attention_weights = []
    for layer in model.transformer_encoder.layers:
        _, weight = layer.self_attn(x, x, x, average_attn_weights=False)
        attention_weights.append(weight.cpu().numpy())
        x = layer(x)
    # (Num_Layers, Batch, Num_Heads, Seq, Seq) -> (Batch, Num_Layers, Num_Heads, Seq, Seq)
    attention_weights = np.asarray(attention_weights).transpose(1, 0, 2, 3, 4)
    return attention_weights

def make_mut_candidate(idx, name, sequence):
    if isinstance(idx, (int, np.integer)):
        idx = [idx]
    
    name_split = name.split(";")
    mut_list = []
    aa_chars = "ACDEFGHIKLMNPQRSTVWY"
    
    if len(name_split) == 1:
        for index in idx:
            original = sequence[index]
            targets = [aa for aa in aa_chars if aa != original]
            for target in targets:
                mut_list.append(f"{name};{original}{index+1}{target}")
    else:
        base_name_dict = {}
        for part in name_split[1:]:
            pos = int(part[1:-1]) - 1
            base_name_dict[pos] = part

        for index in idx:
            if index in base_name_dict:
                continue
            original = sequence[index]
            targets = [aa for aa in aa_chars]
            for target in targets:
                new_dict = base_name_dict.copy()
                new_dict[index] = f"{sequence[index]}{index + 1}{target}"
                sorted_items = sorted(new_dict.items())
                x = name_split[0] + ";" + ";".join([item[1] for item in sorted_items])
                mut_list.append(x)
    return mut_list

def replace_sequence(mut, sequence):
    mut_list = mut.split(";")[1:]
    seq_list = list(sequence)
    for mut_candidate in mut_list:
        try:
            pos = int(mut_candidate[1:-1])
            mut_aa = mut_candidate[-1]
            seq_list[pos-1] = mut_aa
        except ValueError:
            continue
    return "".join(seq_list)

# --- 1. Single Processing Functions (Original Style, Optimized Loading) ---

def tokenize_and_dataloader(name, sequence):
    # Tokenize a single sequence and return a DataLoader
    model, alphabet = get_esm_model()
    batch_converter = alphabet.get_batch_converter()
    
    data = [(name, sequence)]
    batch_labels, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(_DEVICE)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[len(model.layers)], return_contacts=False)
        token_representations = results["representations"][len(model.layers)].squeeze(0) 
    
    # Strip CLS/EOS tokens
    dataloader = DataLoader([[token_representations[1:-1], batch_labels[0]]], batch_size=1)
    return dataloader

def model_prediction(dataloader, model, threshold="2S"):
    # Run inference on a single sequence
    model.eval()
    model.to(_DEVICE)
        
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(_DEVICE)
            
            # Forward (Single)
            wt_prob = model(inputs).cpu().squeeze(0)
            wt_label = (wt_prob >= 0.5).float()
            
            # Attention (Single)
            attention_weights = collect_attention_weights(inputs, model)
        
        # Outlier Detection
        original_idx = listup_outlier_residues(attention_weights, threshold)
        
    return original_idx, wt_prob, wt_label, labels, attention_weights

# --- 2. Batch Processing Functions (Mutation Design Only) ---

def predict_batch_mutations(model, candidates, sequence, batch_size=32, threshold="2S"):
    """
    Batch inference function used only in the mutation design stage.
    Substitution mutations preserve sequence length, so no padding tokens are needed.
    """
    model.eval()
    esm_model, alphabet = get_esm_model()
    batch_converter = alphabet.get_batch_converter()
    
    names = candidates
    seqs = [replace_sequence(c, sequence) for c in candidates]
    
    results_probs = {}
    results_indices = {}
    
    print(f"Processing {len(candidates)} candidates in batches of {batch_size}")
    
    # Process each batch separately with complete cleanup
    total_batches = (len(seqs) + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(0, len(seqs), batch_size),
        total=total_batches,
        desc="Processing batches",
        unit="batch",
    ):
        batch_start = batch_idx
        batch_end = min(batch_idx + batch_size, len(seqs))
        batch_names = names[batch_start:batch_end]
        batch_seqs = seqs[batch_start:batch_end]
        
        try:
            # Force garbage collection before each batch
            gc.collect()
            
            # Step 1: Tokenization (minimal memory)
            batch_data = list(zip(batch_names, batch_seqs))
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(_DEVICE)
            
            # Step 2: ESM Embedding (keep on GPU)
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[len(esm_model.layers)], return_contacts=False)
                token_reprs = results["representations"][len(esm_model.layers)]
                embeddings = token_reprs[:, 1:-1, :]  # Keep on GPU
                
            # Clear all ESM related tensors except embeddings
            del batch_tokens, token_reprs, results
            torch.cuda.empty_cache()
            gc.collect()
            
            # Step 3: Classifier & Attention (directly on GPU)
            with torch.no_grad():
                # Step 4: Classifier prediction
                probs = model(embeddings).cpu()  # Move probs to CPU
                torch.cuda.empty_cache()
                
                # Step 5: Attention weights (using same embeddings on GPU)
                batch_attn = collect_attention_weights_batch(embeddings, model)
                del embeddings  # Clear embeddings after use
                torch.cuda.empty_cache()
                
                # Step 6: Outlier detection
                batch_indices = listup_outlier_residues(batch_attn, threshold)
                del batch_attn
                torch.cuda.empty_cache()
            
            # Step 7: Store results
            for j, name in enumerate(batch_names):
                results_probs[name] = probs[j]
                results_indices[name] = batch_indices[j]
            
            # Clear batch results
            del probs, batch_indices
            torch.cuda.empty_cache()
            gc.collect()
            
            # # Print memory usage
            # if torch.cuda.is_available():
            #     memory_used = torch.cuda.memory_allocated() / 1024**3
            #     memory_reserved = torch.cuda.memory_reserved() / 1024**3
            #     print(f"  GPU Memory: {memory_used:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                tqdm.write(f"OOM at batch {batch_idx//batch_size + 1}. Reducing batch size...")
                torch.cuda.empty_cache()
                gc.collect()
                # Retry remaining candidates with smaller batch size, merging with accumulated results
                smaller_batch_size = max(1, batch_size // 2)
                remaining_probs, remaining_indices = predict_batch_mutations(
                    model, candidates[batch_idx:], sequence,
                    batch_size=smaller_batch_size, threshold=threshold
                )
                results_probs.update(remaining_probs)
                results_indices.update(remaining_indices)
                break
            else:
                raise e
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print("Batch processing completed")
    
    return results_probs, results_indices


# --- Main Logic ---

def scan_switch_mutation(model, sequence, name="unknown", pickle_path=".", max_num_mutation=3, max_num_solution=50, prob_thres=0.5, mode="iter_num", threshold="2S", batch_size=32):
    
    # 1. WT inference: single processing
    wt_dataloader = tokenize_and_dataloader(name, sequence)
    wt_idx, wt_prob, wt_label, wt_name, _ = model_prediction(wt_dataloader, model, threshold)
    
    print(f"The wildtype label probability is ...{wt_prob}")

    convert_dict = {}
    index_dict = {name: np.asarray(wt_idx)}
    results_history = {"No": {}}

    for i in range(max_num_mutation):
        print(f"--- Mutation Step {i+1} Start ---")
        
        # Select parent nodes
        parents = []
        if i == 0:
            parents = [name]
        else:
            if mode == "shortest":
                prev_no = results_history["No"].get(i-1, {})
                if not prev_no: break
                sorted_parents = sorted(prev_no.items(), key=lambda x: x[1], reverse=True)
                parents = [sorted_parents[0][0]]
            else:
                parents = [k for k in index_dict.keys() if len(k.split(";")) == i + 1]

        if not parents:
            break

        # Generate mutation candidates
        all_candidates = []
        for parent in parents:
            if parent not in index_dict: continue
            new_muts = make_mut_candidate(index_dict[parent], parent, sequence)
            for m in new_muts:
                if m not in index_dict:
                    all_candidates.append(m)
        
        all_candidates = sorted(list(set(all_candidates)))
        if not all_candidates:
            break
            
        print(f"Generated {len(all_candidates)} candidates. Predicting in batches...")

        # 2. Mutation screening: batch processing
        # Substitutions preserve length, so batching is safe without padding
        batch_probs, batch_indices = predict_batch_mutations(model, all_candidates, sequence, batch_size=batch_size, threshold=threshold)

        # Aggregate results
        results = {"Convert": {}, "No": {}}
        results["No"][i] = {}
        converted_count = 0
        
        for mut_name in all_candidates:
            if mut_name not in batch_probs:
                continue
            prob = batch_probs[mut_name]
            idx = batch_indices[mut_name]
            mut_label = (prob >= 0.5).float()
            
            # Move indices to CPU to prevent GPU memory accumulation
            index_dict[mut_name] = idx
            
            is_converted = (wt_label == mut_label).sum().item() == 0
            
            if is_converted:
                convert_dict[mut_name] = prob.cpu().numpy() if isinstance(prob, torch.Tensor) else prob
                results["Convert"][mut_name] = prob.cpu().numpy() if isinstance(prob, torch.Tensor) else prob
                converted_count += 1
            else:
                target_prob = 0.0
                if (wt_label == torch.tensor([1,0]).to(wt_label.device)).sum().item() == 2:
                    target_prob = float(prob[1])
                else:
                    target_prob = float(prob[0])
                
                if mode == "iter_prob":
                    results["No"][i][mut_name] = prob.cpu() if isinstance(prob, torch.Tensor) else prob
                else:
                    results["No"][i][mut_name] = target_prob

        print(f"Step {i+1} finished. Converted: {converted_count}")
        
        # Clear GPU memory after each step
        torch.cuda.empty_cache()
        
        with open(f"{pickle_path}/{name}_{mode}_mutation_{i+1}.pkl", "wb") as f:
            pkl.dump(results, f)
        
        results_history["No"][i] = results["No"][i]

        if converted_count > 0 and (mode == "shortest" or mode == "iter_num"):
            print(f"Mutation found. Stopping.")
            break

    # Build final result DataFrame
    if not convert_dict:
        print("The mutation was not found...")
        return None

    keys = list(convert_dict.keys())
    values = list(convert_dict.values())
    if len(values) > 0: values = np.vstack(values)
        
    df = pd.DataFrame(values, index=keys, columns=["NAD", "NADP"])
    
    if (wt_label == torch.tensor([1,0]).to(wt_label.device)).sum().item() == 2:
        df = df.sort_values(by="NADP", ascending=False)
    else:
        df = df.sort_values(by="NAD", ascending=False)
        
    if len(df) > max_num_solution:
        df = df.iloc[:max_num_solution]
        
    return df

def make_df_sorting_by_prob(candidate, wt_label):
    # Legacy compatibility wrapper
    if isinstance(wt_label, torch.Tensor): wt_label = wt_label.cpu().numpy()
    if (wt_label == np.array([1, 0])).all(): label = "NAD"
    else: label = "NADP"

    index, prob = [], []
    for i in range(len(candidate)):
        index.append(candidate[i][0])
        prob.append(candidate[i][1])

    df = pd.DataFrame(prob, columns=["NAD", "NADP"], index=index)
    if label == "NAD": df = df.sort_values(by=["NADP"], ascending=False)
    elif label == "NADP": df = df.sort_values(by=["NAD"], ascending=False)
    return df

def make_max_attention_map(attention_weights):
    max_attn = np.max(np.max(attention_weights, axis=-1), axis=-1)
    plt.figure(figsize=(10,4))
    sns.heatmap(max_attn, cmap="Blues")

def plot_attention_sum(attention_weights, sequence, threshold="2S"):
    att_sum = np.sum(np.sum(np.sum(attention_weights, axis=0), axis=0), axis=0)
    threshold_value = specify_threshold(att_sum, threshold)
    idx = np.where((att_sum > threshold_value) == True)[0]
    
    plt.plot(np.arange(1, len(att_sum) + 1), att_sum)
    plt.plot((1, len(att_sum) + 1), (threshold_value, threshold_value), color="red", linestyle="--")
    print(f"The maximum attention sum is ... {np.max(att_sum):.3f}")
    print(f"The threshold was ... {threshold}")
    
    outlier_residues = []
    for res in idx:
        outlier_residues.append(sequence[res] + str(res+1))
    print(f"The outlier residues are ... {outlier_residues}")
    for i in range(len(outlier_residues)):
        print(f"The attention sum of {outlier_residues[i]} is ... {att_sum[idx[i]]:.3f}")
