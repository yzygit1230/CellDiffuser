import os
from data.pix2pix_dataset import Pix2pixDataset

bags_train_0 = ['Patient1','Patient2','Patient3','Patient4','Patient5','Patient6','Patient7']
bags_train_1 = ['Patient8','Patient9','Patient10','Patient11','Patient12','Patient13','Patient14',
                'Patient15','Patient16','Patient17','Patient18','Patient19','Patient20','Patient21',
                'Patient22','Patient23','Patient24','Patient25']
bags_inference_0 = ['Patient26','Patient27','Patient28']
bags_inference_1 = ['Patient29','Patient30','Patient31','Patient32','Patient33']

class ImmuneDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        baseroot = opt.baseroot
        mode = opt.immune_mode 

        sub_dirs = [d for d in os.listdir(baseroot) if os.path.isdir(os.path.join(baseroot, d))]
        sub_dirs.sort()  

        c_image_paths = []
        s_image_paths = []

        if mode == 'train':
            bags_0 = bags_train_0
            bags_1 = bags_train_1
        elif mode == 'inference':
            bags_0 = bags_inference_0
            bags_1 = bags_inference_1
        else:
            raise ValueError("Invalid mode. Use 'train' or 'inference'.")

        for sub_dir in sub_dirs:
            bags = bags_0 if sub_dir == '0' else bags_1

            for bag in bags:
                croot = os.path.join(baseroot, sub_dir, bag, 'inten/')
                sroot = os.path.join(baseroot, sub_dir, bag, 'pha/')   

                c_files = sorted([f for f in os.listdir(croot) if f.endswith('.tiff')])
                s_files = sorted([f for f in os.listdir(sroot) if f.endswith('.tiff')])

                common_files = set(c_files) & set(s_files)
                if not common_files:
                    print(f"Warning: No matching .tif files found in {croot} and {sroot}. Skipping {bag}.")
                    continue

                for filename in sorted(common_files):
                    c_image_paths.append(os.path.join(croot, filename))
                    s_image_paths.append(os.path.join(sroot, filename))

        self.A_paths = c_image_paths 
        self.B_paths = s_image_paths 

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        instance_paths = [] 
        return c_image_paths, s_image_paths, instance_paths
