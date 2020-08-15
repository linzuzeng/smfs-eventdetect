import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
receptive_field = 224


class Sequence(nn.Module):
    def rolling_window(self, x, window_size, step_size=1):
        # unfold dimension to make sliding windows
        return x.unfold(0, window_size, step_size)

    def __init__(self, input_size,drop_out=0.3):
        super(Sequence, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(p=drop_out),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        """ self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ) """
        self.invregressor = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(512 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(512 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), out.size(1)*out.size(2))

        invregress = self.invregressor(out)
        regress = self.regressor(out)
        return invregress, regress

