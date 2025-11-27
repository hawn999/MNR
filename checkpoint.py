import os
import torch
import shutil

def save_checkpoint(state, is_best, epoch, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), 
                        os.path.join(save_path, 'model_best.pth.tar'))
    

def load_checkpoint(args, model, optimizer=None, verbose=True):

    checkpoint = torch.load(args.resume)

    start_epoch = 0
    best_acc = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_acc" in checkpoint:
        best_acc = checkpoint['best_acc']


    # # Access the full state_dict from the checkpoint
    # full_state_dict = checkpoint["state_dict"]
    #
    # # Print all keys and tensor shapes in the state_dict
    # print(f"Found {len(full_state_dict)} weights in state_dict:\n")
    # for k, v in full_state_dict.items():
    #     print(f"{k}: {v.shape}")

    # ===============================load encoders==============================================
    # # Filter only encoder weights (assuming they are named 'res0', 'res1', etc.)
    # encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if "res" in k}
    # # Check if encoder weights exist
    # assert len(encoder_weights) > 0, "No encoder weights found in checkpoint! Check if 'res' layers exist."
    # # Print encoder layer names being loaded
    # print(f"Found {len(encoder_weights)} encoder weights: {list(encoder_weights.keys())[:5]}...")
    # # Load encoder weights only
    # model.load_state_dict(encoder_weights, strict=False)
    # ==========================================================================================

    # ===============================load entire model==========================================
    # Load the entire state_dict if you want to load all weights
    model.load_state_dict(checkpoint['state_dict'], False)
    # print("Checkpoint Optimizer Parameter Groups:")
    # print(checkpoint["optimizer"]["param_groups"])
    # ==========================================================================================
    # if optimizer is not None and "optimizer" in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(args.device)


    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
                .format(args.resume, start_epoch))
    
    return model, optimizer, best_acc, start_epoch