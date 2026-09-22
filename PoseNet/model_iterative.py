"""
Part 2b — the model, iterative version: the 49 numbers are REFINED rather than
guessed in one shot.

    python model_iterative.py   # builds it, prints parameter count and shapes

    prepare.py      ran once, wrote data/<clip>/crops/*.jpg + labels.npz
    dataset.py      serves (image, context, target)
    model_*.py      image + context -> 49 numbers                <- you are here
    losses.py       how wrong are those 49
    train.py        the loop

EXACTLY ONE THING differs from model_spatial.py: `SpatialHead` becomes
`IterativeHead`. The backbone, the 1x1 bottleneck, the flatten, the rotation
maths and the output format are all the same — so if this scores better,
refinement is the only thing that can have caused it. Same arguments, same
output keys, so train.py swaps between them by changing one word.

The idea: predicting a CORRECTION is easier than predicting an answer
---------------------------------------------------------------------
A single linear layer has to map image features straight onto 49 numbers, in
one shot, getting all of them right simultaneously. That is a hard function.

Iterative error feedback — the approach HMR introduced for exactly this problem
— turns it into a sequence of easier questions. Start from the average pose.
Show the network both the picture AND its own current guess, and ask only
"what is wrong with this?". Add the answer. Repeat.

    guess = average pose
    three times:
        correction = MLP( spatial features, context, guess )
        guess      = guess + correction

WHY IT HELPS, CONCRETELY. The network gets to SEE its own guess. "The left arm
is currently by the side, but the picture shows it raised" is a much easier
judgement than "the left shoulder angle is 1.34 radians" — the first is a
comparison, the second is absolute measurement from scratch. Refinement also
lets a later step fix what an earlier one got wrong, which a single forward
pass cannot do.

The MLP's weights are SHARED across the three steps: one function applied
repeatedly, not three stages.

    image (B,3,256,256)
        |
        v  FeatureExtractor                          (imported)
    features (B,512,8,8)
        |
        v  1x1 conv -> 32 channels, flatten
    (B,2048)  (+) context (B,3)  (+) the current guess (B,49)
        |
        v  IterativeHead, three passes               <- THE ONLY CHANGE
    (B,49)
"""

import torch
import torch.nn as nn

# average_target is re-exported so train.py can swap model files with one import.
from model_spatial import (N_ANGLES, N_OUTPUTS, N_ROT6D, N_SCALES, FeatureExtractor,
                           average_target, rotation_from_6d, starting_guess)

REFINEMENT_STEPS = 3         # HMR's number; gains flatten out after about three


class IterativeHead(nn.Module):
    """Features + context -> 49 numbers, refined from the average pose in N steps.

    The 1x1 bottleneck and flatten are identical to model_spatial.SpatialHead.
    What changes is that the flattened map now feeds an MLP run three times,
    with its own output fed back in each time.

    THE STARTING POSE IS A BUFFER, NOT A PARAMETER: saved in the checkpoint and
    moved to the GPU with the model, but the optimiser never touches it.

    THE LAST LAYER IS ZEROED — weights AND bias — so the first correction is
    exactly zero and the model's very first output is precisely the average
    pose. Same effect as SpatialHead's bias trick, but it falls out more
    naturally here: the thing zeroed is a CORRECTION, and "no correction yet" is
    the obvious initial state.
    """

    def __init__(self, in_channels, filters=32, grid=8, hidden=1024, dropout=0.2,
                 steps=REFINEMENT_STEPS, average=None):
        super().__init__()
        self.steps = steps
        self.reduce = nn.Conv2d(in_channels, filters, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(filters * grid * grid + 3 + N_OUTPUTS, hidden)   # +49: the guess
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, N_OUTPUTS)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        # (1, 49) so it broadcasts over the batch.
        self.register_buffer("start", starting_guess(average)[None, :])

    def forward(self, features, context):
        spatial = self.dropout(self.relu(self.reduce(features)).flatten(1))   # (B, 2048)

        # `expand` rather than `repeat`: a view over the same memory, no copy.
        guess = self.start.expand(features.shape[0], -1)

        for _ in range(self.steps):
            # Show the network the picture AND what it currently believes.
            x = torch.cat([spatial, context, guess], dim=1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            guess = guess + self.fc3(x)        # not in-place: autograd needs each step

        return guess


class PoseNet(nn.Module):
    """One crop + context -> 49 numbers. As model_spatial, but refined.

        backbone    "resnet18" (default), "resnet34" or "resnet50"
        pretrained  start from ImageNet weights
        average     (49,) mean target, for the starting pose. See average_target.
        steps       how many refinement passes

    A drop-in replacement for model_spatial.PoseNet — same arguments, same
    output keys.
    """

    def __init__(self, backbone="resnet18", pretrained=True, average=None,
                 steps=REFINEMENT_STEPS):
        super().__init__()
        self.backbone = FeatureExtractor(backbone, pretrained)
        self.head = IterativeHead(self.backbone.out_channels, steps=steps, average=average)

    def forward(self, image, context):
        features = self.backbone(image)                       # (B, C, 8, 8)
        out = self.head(features, context)                    # (B, 49) after refinement

        angles = out[:, :N_ANGLES]
        scales = out[:, N_ANGLES:N_ANGLES + N_SCALES]
        rot6d, rotation = rotation_from_6d(
            out[:, N_ANGLES + N_SCALES:N_ANGLES + N_SCALES + N_ROT6D])
        camera = out[:, N_ANGLES + N_SCALES + N_ROT6D:]       # (B, 3) — u, v, size

        # Assembled in dataset.target's order, so the loss is one subtraction.
        vector = torch.cat([angles, scales, rot6d, camera], dim=1)

        return {
            'vector': vector,
            'angles': angles,
            'scales': scales,
            'rot6d': rot6d,
            'camera': camera,
            'rotation': rotation,
        }


if __name__ == "__main__":
    net = PoseNet(pretrained=False)
    out = net(torch.zeros(2, 3, 256, 256), torch.zeros(2, 3))
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, "
          f"{REFINEMENT_STEPS} refinement steps")
    for name, value in out.items():
        print(f"  {name:9s} {tuple(value.shape)}")
