import torch
import torch.nn as nn


class CNNBaseLine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=(3, 3), stride=1)

        self.conv2 = nn.Conv2d(5, 7, kernel_size=(3, 3), stride=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        pass


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=1, padding=1)

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)

        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)

        self.conv9_3 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1, padding=0)

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x1 = self.activation(self.conv1_2(self.activation(self.conv1_1(x))))
        x2 = self.activation(self.conv2_2(self.activation(self.conv2_1(self.pool(x1)))))
        x3 = self.activation(self.conv3_2(self.activation(self.conv3_1(self.pool(x2)))))
        x4 = self.activation(self.conv4_2(self.activation(self.conv4_1(self.pool(x3)))))
        x5 = self.activation(self.conv5_2(self.activation(self.conv5_1(self.pool(x4)))))

        x6 = self.activation(
            self.conv6_2(
                self.activation(
                    self.conv6_1(torch.concat((self.up_conv1(x5), x4), dim=1))
                )
            )
        )
        x7 = self.activation(
            self.conv7_2(
                self.activation(
                    self.conv7_1(torch.concat((self.up_conv2(x6), x3), dim=1))
                )
            )
        )
        x8 = self.activation(
            self.conv8_2(
                self.activation(
                    self.conv8_1(torch.concat((self.up_conv3(x7), x2), dim=1))
                )
            )
        )
        return self.conv9_3(
            self.activation(
                self.conv9_2(
                    self.activation(
                        self.conv9_1(torch.concat((self.up_conv4(x8), x1), dim=1))
                    )
                )
            )
        )
