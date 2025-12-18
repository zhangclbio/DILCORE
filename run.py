import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import os
import csv
import warnings
from datetime import datetime
import utils2
import utils
import load_data
from model import DILCR, loss_funcation

warnings.filterwarnings("ignore")
DATASET_PATH = "/home/"
# seed = np.random.randint(0, 10000)  # Generate different random seeds
seed = 123456
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    cancer_type = "kirc"
    # Merge duplicate conf definitions
    conf = dict()
    conf['dataset'] = cancer_type
    conf['view_num'] = 3
    conf['batch_size'] = 128
    conf['encoder_dim'] = [1024]
    conf['feature_dim'] = 512
    conf['peculiar_dim'] = 128
    conf['common_dim'] = 128
    conf['mu_logvar_dim'] = 10
    conf['cluster_var_dim'] = 3 * conf['common_dim']
    conf['up_and_down_dim'] = 512
    conf['use_cuda'] = True
    conf['stop'] = 1e-6
    eval_epoch = 500
    lmda_list = dict()
    lmda_list['rec_lmda'] = 0.9
    lmda_list['KLD_lmda'] = 0.3
    lmda_list['I_loss_lmda'] = 0.1
    conf['kl_loss_lmda'] = 10
    conf['update_interval'] = 50
    conf['lr'] = 1e-4
    conf['pre_epochs'] = 1500
    conf['idec_epochs'] = 500
    
    # Load multi-omics data (Preprocessing method: mean imputation)
    exp, methy, mirna, survival = load_data.load_TCGA(DATASET_PATH+"data/", cancer_type, 'mean')
    
    # Output sample and feature counts
    sample_num = exp.shape[1]
    gene_num = exp.shape[0]
    print(f"Sample count (mRNA): {sample_num}")
    print(f"Gene count (mRNA): {gene_num}")
    
    methy_sample_num = methy.shape[1]
    methy_num = methy.shape[0]
    print(f"Sample count (methylation): {methy_sample_num}")
    print(f"Methylation site count: {methy_num}")
    
    # Fix: Correct miRNA dimension reference
    mirna_sample_num = mirna.shape[1]
    mirna_num = mirna.shape[0]
    print(f"Sample count (miRNA): {mirna_sample_num}")
    print(f"miRNA count: {mirna_num}")     
    
    # Convert to tensor and move to device
    exp_df = torch.tensor(exp.values.T, dtype=torch.float32).to(device)
    methy_df = torch.tensor(methy.values.T, dtype=torch.float32).to(device)
    mirna_df = torch.tensor(mirna.values.T, dtype=torch.float32).to(device)
    full_data = [utils2.p_normalize(exp_df), utils2.p_normalize(methy_df), utils2.p_normalize(mirna_df)]
    
    # Set cluster number based on cancer type
    if conf['dataset'] == "aml":
        conf['cluster_num'] = 3
    elif conf['dataset'] == "brca":
        conf['cluster_num'] = 5
    elif conf['dataset'] == "skcm":
        conf['cluster_num'] = 5
    elif conf['dataset'] == "lihc":
        conf['cluster_num'] = 5
    elif conf['dataset'] == "coad":
        conf['cluster_num'] = 4
    elif conf['dataset'] == "kirc":
        conf['cluster_num'] = 4
    elif conf['dataset'] == "gbm":
        conf['cluster_num'] = 3
    elif conf['dataset'] == "ov":
        conf['cluster_num'] = 3
    elif conf['dataset'] == "lusc":
        conf['cluster_num'] = 3
    elif conf['dataset'] == "sarc":
        conf['cluster_num'] = 5
    
    # Fix random seed setup (only once)
    setup_seed(seed=seed)
    
    # ========================Result File====================
    folder = "/home/result/{}_result".format(conf['dataset'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Create result CSV file
    result = open("{}/{}_{}.csv".format(folder, conf['dataset'], conf['cluster_num']), 'w+')
    writer = csv.writer(result)
    writer.writerow(['p_value', 'log2p', 'log10p', 'epoch', 'training_phase'])
    
    # =======================Initialize Model and Loss Function====================
    in_dim = [exp_df.shape[1], methy_df.shape[1], mirna_df.shape[1]]
    model = DILCR(
        in_dim=in_dim, 
        encoder_dim=conf['encoder_dim'], 
        feature_dim=conf['feature_dim'],
        common_dim=conf['common_dim'],
        mu_logvar_dim=conf['mu_logvar_dim'], 
        cluster_var_dim=conf['cluster_var_dim'],
        up_and_down_dim=conf['up_and_down_dim'], 
        cluster_num=conf['cluster_num'],
        peculiar_dim=conf['peculiar_dim'], 
        view_num=conf['view_num'], 
        device=device
    )
    model = model.to(device=device)
    opt = torch.optim.AdamW(lr=conf['lr'], params=model.parameters())
    loss = loss_funcation()
    
    # =======================Pre-training VAE====================
    print("Pre-training VAE ----------------------- Dataset: {} | Cluster number: {}".format(conf['dataset'], conf['cluster_num']))
    pbar = tqdm(range(conf['pre_epochs']), ncols=120)
    max_log10p = 0.0
    max_label_pretrain = []
    
    for epoch in pbar:
        # Batch training
        sample_num = exp_df.shape[0]
        randidx = torch.randperm(sample_num)
        
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [
                utils2.p_normalize(exp_df[idx]), 
                utils2.p_normalize(methy_df[idx]), 
                utils2.p_normalize(mirna_df[idx])
            ]
            
            # Forward pass
            out_list, latent_dist = model(data_batch)
            
            # Calculate loss
            total_loss, loss_dict = loss(
                view_num=conf['view_num'], 
                data_batch=data_batch, 
                out_list=out_list,
                latent_dist=latent_dist, 
                lmda_list=lmda_list, 
                batch_size=conf['batch_size'], 
                model=model
            )
            
            # Backward pass and optimization
            total_loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Evaluate clustering performance every eval_epoch
        if (epoch + 1) % eval_epoch == 0:
            with torch.no_grad():
                model.eval()
                out_list, latent_dist = model(full_data)
                
                # K-means clustering on latent features
                kmeans = KMeans(
                    n_clusters=conf['cluster_num'], 
                    n_init=20, 
                    random_state=seed, 
                    init="k-means++"
                )
                kmeans.fit(latent_dist['cluster_var'].cpu().numpy())
                pred_labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_
                
                # Survival analysis (log-rank test)
                survival["label"] = np.array(pred_labels)
                logrank_res = utils2.log_rank(survival)
                
                # Write evaluation results
                writer.writerow([
                    logrank_res['p'], 
                    logrank_res['log2p'], 
                    logrank_res['log10p'], 
                    epoch, 
                    "pre_training"
                ])
                result.flush()
                model.train()
            
            # Save best model (based on log10p value)
            if logrank_res['log10p'] > max_log10p:
                max_log10p = logrank_res['log10p']
                max_label_pretrain = pred_labels
                torch.save(
                    model.state_dict(), 
                    "{}/{}_pretrain_best_log10p.pdparams".format(folder, conf['dataset'])
                )
        
        # Update progress bar with loss metrics
        pbar.set_postfix(
            total_loss="{:3.4f}".format(loss_dict['loss'].item()),
            reconstruction_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
            KLD_loss="{:3.4f}".format(loss_dict['KLD'].item()),
            InfoNCE_loss="{:3.4f}".format(loss_dict['I_loss'].item())
        )
    
    # =======================IDEC Clustering Optimization=====================
    print("IDEC training ----------------------- Dataset: {} | Cluster number: {}".format(conf['dataset'], conf['cluster_num']))
    
    # Initialize cluster centers with pre-trained features
    out_list, latent_dist = model(full_data)
    kmeans = KMeans(
        n_clusters=conf['cluster_num'], 
        random_state=seed, 
        init="k-means++"
    ).fit(latent_dist['cluster_var'].detach().cpu().numpy())
    
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    y_pred_last = kmeans.labels_
    max_log10p_idec = 0.0
    max_label_idec = y_pred_last
    
    pbar = tqdm(range(conf['idec_epochs']), ncols=120)
    for epoch in pbar:
        # Update soft assignment every update_interval
        if epoch % conf['update_interval'] == 0:
            with torch.no_grad():
                _, latent_dist = model(full_data)
                
                # Calculate soft assignment (q) and target distribution (p)
                tmp_q = latent_dist['q']
                y_pred = tmp_q.cpu().numpy().argmax(1)
                weight = tmp_q ** 2 / tmp_q.sum(0)
                p_dist = (weight.t() / weight.sum(1)).t()
                
                # Calculate label change rate
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                
                # Survival analysis evaluation
                survival["label"] = np.array(y_pred)
                logrank_res = utils2.log_rank(survival)
                writer.writerow([
                    logrank_res['p'], 
                    logrank_res['log2p'], 
                    logrank_res['log10p'], 
                    epoch, 
                    "IDEC_training"
                ])
                result.flush()
                
                # Save best IDEC model
                if logrank_res['log10p'] > max_log10p_idec:
                    print(f"Updated best log10p: {logrank_res['log10p']}")
                    max_log10p_idec = logrank_res['log10p']
                    max_label_idec = y_pred
                    torch.save(
                        model.state_dict(), 
                        "{}/{}_idec_best_log10p.pdparams".format(folder, conf['dataset'])
                    )
                
                # Early stopping if label change rate is below threshold
                if epoch > 0 and delta_label < conf['stop']:
                    print(f'Label change rate {delta_label:.4f} < tolerance {conf["stop"]}')
                    print('Reached tolerance threshold. Stopping training.')
                    break
                
                # Create visualization directory if not exists
                vis_dir = os.path.dirname(DATASET_PATH + f"result/{conf['dataset']}_result/latent_space_images.png")
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                
                # Visualize latent space (tSNE)
                utils2.visualize_latent_space(
                    latent_dist, 
                    epoch, 
                    save_dir=DATASET_PATH + f"result/{conf['dataset']}_result/latent_space_epoch_{epoch}.png", 
                    title="Latent Space (tSNE)", 
                    method='tSNE',
                    labels=y_pred  # Fix: Use current IDEC labels instead of pretrain labels
                )
                
                # Survival curve analysis
                utils2.lifeline_analysis(
                    survival, 
                    title_g=f"{conf['dataset']} Survival Analysis (Epoch {epoch})", 
                    save_path=DATASET_PATH + f"result/{conf['dataset']}_result/survival_curve_epoch_{epoch}.png"
                )
        
        # Batch training with KL divergence loss
        sample_num = exp_df.shape[0]
        randidx = torch.randperm(sample_num)
        
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [
                utils2.p_normalize(exp_df[idx]), 
                utils2.p_normalize(methy_df[idx]), 
                utils2.p_normalize(mirna_df[idx])
            ]
            
            out_list, latent_dist = model(data_batch)
            
            # KL divergence loss between soft assignment and target distribution
            kl_div_loss = F.kl_div(latent_dist['q'].log(), p_dist[idx])
            
            # Calculate base loss
            total_loss, loss_dict = loss(
                view_num=conf['view_num'], 
                data_batch=data_batch, 
                out_list=out_list,
                latent_dist=latent_dist, 
                lmda_list=lmda_list, 
                batch_size=conf['batch_size'], 
                model=model
            )
            
            # Combine losses (base loss + KL divergence loss)
            total_loss = total_loss + conf['kl_loss_lmda'] * kl_div_loss
            total_loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Update progress bar with loss metrics
        pbar.set_postfix(
            total_loss="{:3.4f}".format(loss_dict['loss'].item()),
            reconstruction_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
            KLD_loss="{:3.4f}".format(loss_dict['KLD'].item()),
            InfoNCE_loss="{:3.4f}".format(loss_dict['I_loss'].item()),
            KL_div_loss="{:3.4f}".format(kl_div_loss.item())
        )
    
    # =======================Post-training Analysis=====================
    # 1. Clinical enrichment analysis (Pretrain best labels)
    survival["label"] = np.array(max_label_pretrain)
    clinical_data = utils2.get_clinical(DATASET_PATH + "data/clinical", survival, conf["dataset"])
    clinical_enrich_pretrain = utils2.clinical_enrichement(clinical_data['label'], clinical_data)
    
    # 2. Clinical enrichment analysis (IDEC best labels)
    survival["label"] = np.array(max_label_idec)
    clinical_data = utils2.get_clinical(DATASET_PATH + "data/clinical", survival, conf["dataset"])
    # Clean clinical data (remove invalid values)
    clinical_data = clinical_data.dropna(subset=['pathologic_M', 'pathologic_N'])
    clinical_data.to_csv(f"{folder}/{conf['dataset']}_clinical_data.csv", index=False)
    clinical_enrich_idec = utils2.clinical_enrichement(clinical_data['label'], clinical_data)
    
    # 3. Save final clustering results
    exp = exp.T
    sample_ids = exp.index   
    idec_clustering_result = pd.DataFrame({
        'Sample_ID': sample_ids,
        'IDEC_Cluster_Label': max_label_idec
    })
    idec_clustering_result.to_csv(f"{folder}/{conf['dataset']}_idec_clustering_result.csv", index=False)
    
    # 4. Print final results
    print(f"{conf['dataset']}:    DILCR-Pretrain:  {clinical_enrich_pretrain}/{max_log10p:.1f}   DILCR-IDEC:   {clinical_enrich_idec}/{max_log10p_idec:.1f}")
    
    # Close result file
    result.close()
