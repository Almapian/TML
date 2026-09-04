"""Reusable modelling machinery shared by the stage 1-3 scripts and notebooks.

    config         seeds, split boundaries, site constants, device selection
    layers         Encoder / BatchedAttention (shared by stages 2 and 3)
    architectures  every nn.Module used by stages 1-3
    windowing      sliding-window construction for each stage
    datasets       loading, splitting, UTide fitting, scaling -> ready-to-train tensors
    training       training loop, masked losses, metrics, checkpoint handling
"""
