class TFPL(nn.Module):

    def __init__(self, dim, num_frames=4):
        super().__init__()

        self.num_frames = num_frames

        self.resblocks = nn.ModuleList([
            ResBlock(dim) for _ in range(num_frames)
        ])

        self.norm = nn.BatchNorm2d(dim)

        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(dim * num_frames, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, bev_seq):
        """
        bev_seq: list of BEV features
        [BEV^{i-n}, ..., BEV^{i-1}]
        each: [B,C,H,W]
        """

        assert len(bev_seq) == self.num_frames

        outputs = []

        x = bev_seq[0]
        x = self.resblocks[0](x)
        outputs.append(x)

        for i in range(1, self.num_frames):
            x = bev_seq[i] + outputs[-1]   # Add
            x = self.norm(x)
            x = self.resblocks[i](x)
            outputs.append(x)

        fused = torch.cat(outputs, dim=1)

        bev_pred = self.fusion_mlp(fused)

        return bev_pred