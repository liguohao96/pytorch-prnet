from torch import nn


def padding_same_conv2d(input_size, in_c, out_c, kernel_size=4, stride=1):
    output_size = input_size // stride
    padding_num = stride * (output_size - 1) - input_size + kernel_size
    if padding_num % 2 == 0:
        return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding_num // 2, bias=False))
    else:
        return nn.Sequential(
            nn.ConstantPad2d((padding_num // 2, padding_num // 2 + 1, padding_num // 2, padding_num // 2 + 1), 0),
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        )

# import threading
class resBlock(nn.Module):
    instance_num = 0
    # instance_num_lock = threading.Lock()
    def __init__(self, in_c, out_c, kernel_size=4, stride=1, input_size=None):
        super().__init__()
        self.instance_idx = self.__class__.instance_num
        self.__class__.instance_num += 1
        assert kernel_size == 4
        self.shortcut = lambda x: x
        self.tf_map = {}
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
            self.tf_map['{}/shortcut/weights'.format(self.instance_name())] = 'shortcut.weight'

        main_layers = [
            nn.Conv2d(in_c, out_c // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
        ]
        self.tf_map['{}/Conv/weights'.format(self.instance_name())] = 'main.0.weight'
        self.tf_map['{}/Conv/BatchNorm/gamma'.format(self.instance_name())] = 'main.1.weight'
        self.tf_map['{}/Conv/BatchNorm/beta'.format(self.instance_name())] = 'main.1.bias'
        self.tf_map['{}/Conv/BatchNorm/moving_mean'.format(self.instance_name())] = 'main.1.running_mean'
        self.tf_map['{}/Conv/BatchNorm/moving_variance'.format(self.instance_name())] = 'main.1.running_var'

        main_layers.extend([
            *padding_same_conv2d(input_size, out_c // 2, out_c // 2, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_c // 2, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)])
        conv_idx = len(main_layers) - 3
        self.tf_map['{}/Conv_1/weights'.format(self.instance_name())] = 'main.{}.weight'.format(conv_idx)
        self.tf_map['{}/Conv_1/BatchNorm/gamma'.format(self.instance_name())] = 'main.{}.weight'.format(conv_idx+1)
        self.tf_map['{}/Conv_1/BatchNorm/beta'.format(self.instance_name())] = 'main.{}.bias'.format(conv_idx+1)
        self.tf_map['{}/Conv_1/BatchNorm/moving_mean'.format(self.instance_name())] = 'main.{}.running_mean'.format(conv_idx+1)
        self.tf_map['{}/Conv_1/BatchNorm/moving_variance'.format(self.instance_name())] = 'main.{}.running_var'.format(conv_idx+1)

        main_layers.extend(
            padding_same_conv2d(input_size, out_c // 2, out_c, kernel_size=1, stride=1)
        )
        conv_idx = len(main_layers) - 1
        self.tf_map['{}/Conv_2/weights'.format(self.instance_name())] = 'main.{}.weight'.format(conv_idx)
        self.main = nn.Sequential(*main_layers)
        self.activate = nn.Sequential(
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )
        self.tf_map['{}/BatchNorm/gamma'.format(self.instance_name())] = 'activate.0.weight'
        self.tf_map['{}/BatchNorm/beta'.format(self.instance_name())]  = 'activate.0.bias'
        self.tf_map['{}/BatchNorm/moving_mean'.format(self.instance_name())]      = 'activate.0.running_mean'
        self.tf_map['{}/BatchNorm/moving_variance'.format(self.instance_name())]  = 'activate.0.running_var'

    def instance_name(self):
        return 'resBlock' if self.instance_idx == 0 else 'resBlock_{}'.format(self.instance_idx)

    def forward(self, x):
        shortcut_x = self.shortcut(x)
        main_x = self.main(x)
        x = self.activate(shortcut_x + main_x)
        return x


class upBlock(nn.Module):
    convtranspose_num = 0
    def __init__(self, in_c, out_c, conv_num=2):
        super().__init__()
        self.tf_map = {}
        additional_conv = []
        layer_length = 4

        self.convtrans_idx = self.__class__.convtranspose_num
        self.__class__.convtranspose_num += 1
        self.tf_map['{}/weights'.format(self.convtranspose_name())] = 'main.0.weight'
        self.tf_map['{}/BatchNorm/gamma'.format(self.convtranspose_name())] = 'main.1.weight'
        self.tf_map['{}/BatchNorm/beta'.format(self.convtranspose_name())] = 'main.1.bias'
        self.tf_map['{}/BatchNorm/moving_mean'.format(self.convtranspose_name())] = 'main.1.running_mean'
        self.tf_map['{}/BatchNorm/moving_variance'.format(self.convtranspose_name())] = 'main.1.running_var'
        for i in range(1, conv_num+1):
            self.convtrans_idx = self.__class__.convtranspose_num
            self.__class__.convtranspose_num += 1
            additional_conv += [
                nn.ConstantPad2d((2, 1, 2, 1), 0),
                nn.ConvTranspose2d(out_c, out_c, kernel_size=4, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
                nn.ReLU(inplace=True)
            ]
            self.tf_map['{}/weights'.format(self.convtranspose_name())] = 'main.{}.weight'.format(i*layer_length + 0)
            self.tf_map['{}/BatchNorm/gamma'.format(self.convtranspose_name())] = 'main.{}.weight'.format(i*layer_length + 1)
            self.tf_map['{}/BatchNorm/beta'.format(self.convtranspose_name())] = 'main.{}.bias'.format(i*layer_length + 1)
            self.tf_map['{}/BatchNorm/moving_mean'.format(self.convtranspose_name())] = 'main.{}.running_mean'.format(i*layer_length + 1)
            self.tf_map['{}/BatchNorm/moving_variance'.format(self.convtranspose_name())] = 'main.{}.running_var'.format(i*layer_length + 1)

        self.main = nn.Sequential(
            # nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),
            *additional_conv
            )

    def convtranspose_name(self):
        return 'Conv2d_transpose' if self.convtrans_idx == 0 else 'Conv2d_transpose_{}'.format(self.convtrans_idx)

    def forward(self, x):
        x = self.main(x)
        return x


class PRNet(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super().__init__()
        size = 16
        self.input_conv = nn.Sequential( #*[
            *padding_same_conv2d(256, in_channel, size, kernel_size=4, stride=1),  # 256x256x16
            nn.BatchNorm2d(size, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
            # ]
        ) 
        self.tf_map = {}
        conv_idx = len(self.input_conv) - 3
        self.tf_map['resfcn256/Conv/weights'] = 'input_conv.{}.weight'.format(conv_idx)
        self.tf_map['resfcn256/Conv/BatchNorm/gamma'] = 'input_conv.{}.weight'.format(conv_idx + 1)
        self.tf_map['resfcn256/Conv/BatchNorm/beta'] = 'input_conv.{}.bias'.format(conv_idx + 1)
        self.tf_map['resfcn256/Conv/BatchNorm/moving_mean'] = 'input_conv.{}.running_mean'.format(conv_idx + 1)
        self.tf_map['resfcn256/Conv/BatchNorm/moving_variance'] = 'input_conv.{}.running_var'.format(conv_idx + 1)
        self.down_conv_1 = resBlock(size, size * 2, kernel_size=4, stride=2, input_size=256)  # 128x128x32
        self.down_conv_2 = resBlock(size * 2, size * 2, kernel_size=4, stride=1, input_size=128)  # 128x128x32
        self.down_conv_3 = resBlock(size * 2, size * 4, kernel_size=4, stride=2, input_size=128)  # 64x64x64
        self.down_conv_4 = resBlock(size * 4, size * 4, kernel_size=4, stride=1, input_size=64)  # 64x64x64
        self.down_conv_5 = resBlock(size * 4, size * 8, kernel_size=4, stride=2, input_size=64)  # 32x32x128
        self.down_conv_6 = resBlock(size * 8, size * 8, kernel_size=4, stride=1, input_size=32)  # 32x32x128
        self.down_conv_7 = resBlock(size * 8, size * 16, kernel_size=4, stride=2, input_size=32)  # 16x16x256
        self.down_conv_8 = resBlock(size * 16, size * 16, kernel_size=4, stride=1, input_size=16)  # 16x16x256
        self.down_conv_9 = resBlock(size * 16, size * 32, kernel_size=4, stride=2, input_size=16)  # 8x8x512
        self.down_conv_10 = resBlock(size * 32, size * 32, kernel_size=4, stride=1, input_size=8)  # 8x8x512

        self.center_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size * 32, size * 32, kernel_size=4, stride=1, padding=3, bias=False),  # 8x8x512
            nn.BatchNorm2d(size * 32, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )
        self.tf_map['resfcn256/Conv2d_transpose/weights'] = 'center_conv.1.weight'
        self.tf_map['resfcn256/Conv2d_transpose/BatchNorm/gamma'] = 'center_conv.2.weight'
        self.tf_map['resfcn256/Conv2d_transpose/BatchNorm/beta'] = 'center_conv.2.bias'
        self.tf_map['resfcn256/Conv2d_transpose/BatchNorm/moving_mean'] = 'center_conv.2.running_mean'
        self.tf_map['resfcn256/Conv2d_transpose/BatchNorm/moving_variance'] = 'center_conv.2.running_var'
        upBlock.convtranspose_num = 1

        self.up_conv_5 = upBlock(size * 32, size * 16)  # 16x16x256
        self.up_conv_4 = upBlock(size * 16, size * 8)  # 32x32x128
        self.up_conv_3 = upBlock(size * 8, size * 4)  # 64x64x64

        self.up_conv_2 = upBlock(size * 4, size * 2, 1)  # 128x128x32
        self.up_conv_1 = upBlock(size * 2, size, 1)  # 256x256x16

        convtranspose_idx = upBlock.convtranspose_num
        self.output_conv = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(size, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True),

            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.001),
            nn.Sigmoid()
        )
        self.tf_map['resfcn256/Conv2d_transpose_{}/weights'.format(convtranspose_idx)] = 'output_conv.1.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/gamma'.format(convtranspose_idx)] = 'output_conv.2.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/beta'.format(convtranspose_idx)] = 'output_conv.2.bias'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_mean'.format(convtranspose_idx)] = 'output_conv.2.running_mean'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_variance'.format(convtranspose_idx)] = 'output_conv.2.running_var'

        convtranspose_idx += 1
        self.tf_map['resfcn256/Conv2d_transpose_{}/weights'.format(convtranspose_idx)] = 'output_conv.5.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/gamma'.format(convtranspose_idx)] = 'output_conv.6.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/beta'.format(convtranspose_idx)] = 'output_conv.6.bias'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_mean'.format(convtranspose_idx)] = 'output_conv.6.running_mean'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_variance'.format(convtranspose_idx)] = 'output_conv.6.running_var'

        convtranspose_idx += 1
        self.tf_map['resfcn256/Conv2d_transpose_{}/weights'.format(convtranspose_idx)] = 'output_conv.9.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/gamma'.format(convtranspose_idx)] = 'output_conv.10.weight'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/beta'.format(convtranspose_idx)] = 'output_conv.10.bias'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_mean'.format(convtranspose_idx)] = 'output_conv.10.running_mean'
        self.tf_map['resfcn256/Conv2d_transpose_{}/BatchNorm/moving_variance'.format(convtranspose_idx)] = 'output_conv.10.running_var'
        self.collact_names()
    
    def collact_names(self):
        for name, child in self.named_children():
            if hasattr(child, 'tf_map'):
                child_map = getattr(child, 'tf_map')
                for k, v in child_map.items():
                    self.tf_map['resfcn256/{}'.format(k)] = '{}.{}'.format(name, v)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_conv_1(x)
        x = self.down_conv_2(x)
        x = self.down_conv_3(x)
        x = self.down_conv_4(x)
        x = self.down_conv_5(x)
        x = self.down_conv_6(x)
        x = self.down_conv_7(x)
        x = self.down_conv_8(x)
        x = self.down_conv_9(x)
        x = self.down_conv_10(x)

        x = self.center_conv(x)

        x = self.up_conv_5(x)
        x = self.up_conv_4(x)
        x = self.up_conv_3(x)
        x = self.up_conv_2(x)
        x = self.up_conv_1(x)
        x = self.output_conv(x)
        return x
