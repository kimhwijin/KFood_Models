class KerasInceptionResNetV2_Weights():
    def __init__(self):
        self.STEM = {
            'conv0': 32,
            'conv1': 32,
            'conv2': 64,
            'conv4': 80,
            'conv5': 192,
            'branch0': 96,
            'branch1_0': 48,
            'branch1_1': 64,
            'branch2_0': 64,
            'branch2_1': 96,
            'branch2_2': 96,
            'branch_pool_2': 64
        }
        self.BLOCK35 = {
            'branch0': 32,
            'branch1_0': 32,
            'branch1_1': 32,
            'branch2_0': 32,
            'branch2_1': 48,
            'branch2_2': 64,
        }

        self.BLOCK17 = {
            'branch0': 192,
            'branch1_0': 128,
            'branch1_1': 160,
            'branch1_2': 192,
        }
        self.BLOCK8 = {
            'branch0': 192,
            'branch1_0': 192,
            'branch1_1': 224,
            'branch1_2': 256
        }
        self.REDUCTION_A = {
            'branch0': 384,
            'branch1_0': 256,
            'branch1_1': 256,
            'branch1_2': 384,
        }
        self.REDUCTION_B = {
            'branch0_0': 256,
            'branch0_1': 384,
            'branch1_0': 256,
            'branch1_1': 288,
            'branch2_0': 256,
            'branch2_1': 288,
            'branch2_2': 320,
        }


