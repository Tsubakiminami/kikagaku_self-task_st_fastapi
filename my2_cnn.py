# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# ネットワーク定義を行うモジュール

# 必要なモジュールをインポートする
import pytorch_lightning as pl      # Net1
import torch.nn as nn               # Net2, Net3
import torch.nn.functional as F


class Net1(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # ここでのオブジェクト定義が前回と変わる
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)  # グレースケールなので in_channels=1とした。　出来上がりサイズを揃えたいので padding=1、kernel_size=3で3X3のフィルタになる
        self.bn = nn.BatchNorm2d(3) # 画像なのでBatchNorm2dを選択、out_channels=3としたので 3とする。
        self.fc = nn.Linear(588, 10)    # 3 X 14 X 14 =588


    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 588)
        h = self.fc(h)
        return h


    # def training_step(self, batch, batch_idx):
    #     x, t = batch
    #     y = self(x)
    #     loss = F.cross_entropy(y, t)
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
    #     return loss


    # def validation_step(self, batch, batch_idx):
    #     x, t = batch
    #     y = self(x)
    #     loss = F.cross_entropy(y, t)
    #     self.log('val_loss', loss, on_step=False, on_epoch=True)
    #     self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1), on_step=False, on_epoch=True)
    #     return loss


    # def test_step(self, batch, batch_idx):
    #     x, t = batch
    #     y = self(x)
    #     loss = F.cross_entropy(y, t)
    #     self.log('test_loss', loss, on_step=False, on_epoch=True)
    #     self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=10, top_k=1), on_step=False, on_epoch=True)
    #     return loss


    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    #     return optimizer

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x
