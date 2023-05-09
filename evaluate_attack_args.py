from functools import partial



from metrics.ranking_metrics import *
from trainers import *
from trainers.losses import *

data_path = "./data/gowalla"  # # Dataset path and loader
use_cuda = True  # If using GPU or CPU
seed = 1  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}

""" Victim model hyper-parameters."""
vict_mf = {
    **shared_params,
    "epochs":50,
    "lr": 1e-5,
    "l2":1e-5,
    "save_feq": 50,
    "batch_size": 128,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": MFTrainer,
        "model_name": "MF",
        "model_type": "MF",
        "hidden_dims": [128],
        "optim_method": "sgd"

    }
}

vict_bprmf = {
    **shared_params,
    "epochs":100,
    "lr": 0.001,
    "l2":0,
    "save_feq": 100,
    "batch_size": 256,
    "valid_batch_size": 32,
    "model": {
        "trainer_class": BPRMFTrainer,
        "model_name": "BPRMF",
        "model_type": "BPRMF",
        "hidden_dims": 32,
    }
}

vict_dmf = {
    **shared_params,
    "epochs": 10,
    "lr": 0.00001,
    "save_feq": 10,
    "batch_size": 256,
    "valid_batch_size": 100,
    "model": {
        "trainer_class": DMFTrainer,
        "model_name": "DMF",  
        "model_type": "DMF",   
        "userlayers": [512,64], 
        "itemlayers": [1024,64], 
    }
}

vict_itemcf = {
    **shared_params,
    "epochs": 1,
    "lr": 0.0,
    "l2": 0.0,
    "save_feq": 1,
    "batch_size": 128,
    "valid_batch_size": 128,
    "model": {
        "trainer_class": ItemCFTrainer,
        "model_name": "ItemCF",
        "knn": 50
    }
}
vict_cml = {
    **shared_params,
    "epochs": 100,
    "lr": 1e-3,
    "l2": 1e-4,
    "save_feq": 100,
    "batch_size": 512,
    "valid_batch_size": 32,
    "model": {
        "trainer_class": NCFTrainer,
        "model_name": "CML",
        "model_type": "CML",
        "num_factors": 256
    }
}
vict_ncf = {
    **shared_params,
    "epochs": 90,
    "lr": 3e-4,
    "l2": 1e-4,
    "save_feq": 90,
    "batch_size": 512,
    "valid_batch_size": 32,
    "model": {
        "trainer_class": NCFTrainer,
        "model_name": "NeuralCF",
        "model_type": "NeuralCF",
        "hidden_dims": [128],
        "num_factors": 256,
        "ac": "tanh",
    }
}
vict_wmf_sgd = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-2,
    "l2": 1e-5,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "WeightedMF-sgd",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}
vict_user_ae = {
    **shared_params,
    "epochs": 70,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 70,
    "batch_size": 1024,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": UserAETrainer,
        "model_name": "UserVAE",
        "model_type": "UserVAE",
        "hidden_dims": [512, 256],
        "betas": [0.0, 1e-4, 1.0],
        "recon_loss": mult_ce_loss
    }
}
vict_item_ae = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": ItemAETrainer,
        "model_name": "ItemAE",
        "hidden_dims": [256, 128],
        "recon_loss": partial(mse_loss, weight=20)
    }
}

""" Attack evaluation hyper-parameters."""
attack_eval_args = {
    **shared_params,

    # Path to the fake data.
    # If None, then evaluate clean performance without attack.
    "fake_data_path": "./outputs/Sur-WeightedMF-sgd_fake_data_best.npz",
    # Path to the target items.
    "target_items_path": "./outputs/sampled_target_items_5_head.npz",

    # List of victim models to evaluate.
    # "victims": [vict_itemcf, vict_wmf_sgd, vict_item_ae,vict_user_ae]
    # "victims": [vict_ncf,vict_cml,vict_bprmf]

}
