# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import os
import torch
import time
from deit import deit_tiny_patch16_224
from videomamba import videomamba_tiny
torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60


def replace_layernorm(net):
    import apex
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(net, child_name, apex.normalization.FusedLayerNorm(
                child.weight.size(0)))
        else:
            replace_layernorm(child)


def compute_throughput_cpu(name, model, device, batch_size, frame=8, resolution=224):
    inputs = torch.randn(batch_size, 3, frame, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(), 'images/s @ batch_size', batch_size, flush=True)


def compute_throughput_cuda(name, model, device, batch_size, frame=8, resolution=224):
    inputs = torch.randn(batch_size, 3, frame, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(), 'images/s @ batch_size', batch_size, flush=True)
    

def compute_throughput_and_memory_usage_cuda(name, model, device, batch_size, frame=8, resolution=224):
    inputs = torch.randn(batch_size, 3, frame, resolution, resolution, device=device)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    initial_mem_allocated = torch.cuda.memory_allocated(device)
    initial_mem_reserved = torch.cuda.memory_reserved(device)

    start = time.time()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        while time.time() - start < T0:
            model(inputs)
    torch.cuda.synchronize()

    final_mem_allocated = torch.cuda.memory_allocated(device)
    final_mem_reserved = torch.cuda.memory_reserved(device)
    mem_allocated_diff = final_mem_allocated - initial_mem_allocated
    mem_reserved_diff = final_mem_reserved - initial_mem_reserved

    timing = []
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    
    timing = torch.as_tensor(timing, dtype=torch.float32)
    throughput = batch_size / timing.mean().item()
    
    print(f"{name} {device} {throughput} images/s @ batch size {batch_size}", flush=True)
    print(f"Memory Allocated Increase: {mem_allocated_diff / (1024**2):.2f} MB", flush=True)
    print(f"Memory Reserved Increase: {mem_reserved_diff / (1024**2):.2f} MB", flush=True)


# for device in ['cuda:0', 'cpu']:
for device in ['cuda:0']:
    print("For evaluation, pleaset set `rms_norm=False` and `fused_add_norm=False`")

    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread', flush=True)
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        # compute_throughput = compute_throughput_cuda
        compute_throughput = compute_throughput_and_memory_usage_cuda

    for model_name, batch_size0, frame, resolution in [
        ('deit_tiny_patch16_224', 128, 8, 224),
        ('deit_tiny_patch16_224', 128, 16, 224),
        ('deit_tiny_patch16_224', 64, 24, 224),
        ('deit_tiny_patch16_224', 32, 32, 224),
        ('deit_tiny_patch16_224', 32, 40, 224),
        ('deit_tiny_patch16_224', 24, 48, 224),
        ('deit_tiny_patch16_224', 16, 56, 224),
        ('deit_tiny_patch16_224', 12, 64, 224),
        ('videomamba_tiny', 128, 8, 224),
        ('videomamba_tiny', 128, 16, 224),
        ('videomamba_tiny', 128, 24, 224),
        ('videomamba_tiny', 128, 32, 224),
        ('videomamba_tiny', 128, 40, 224),
        ('videomamba_tiny', 128, 48, 224),
        ('videomamba_tiny', 128, 56, 224),
        ('videomamba_tiny', 128, 64, 224),
    ]:

        if device == 'cpu':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()
        inputs = torch.randn(batch_size, 3, frame, resolution, resolution, device=device)
        model = eval(model_name)(img_size=resolution, num_frames=frame, num_classes=400)
        replace_layernorm(model)
        model.to(device)
        model.eval()
        model = torch.jit.trace(model, inputs)
        compute_throughput(model_name, model, device, batch_size, frame=frame, resolution=resolution)

