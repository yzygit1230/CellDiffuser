import os
from data.pix2pix_dataset import Pix2pixDataset


class ColorizationDataset(Pix2pixDataset):
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
        sub_dirs = [d for d in os.listdir(baseroot) if os.path.isdir(os.path.join(baseroot, d))]
        sub_dirs.sort() 
        c_image_paths = []
        s_image_paths = []

        for sub_dir in sub_dirs:
            croot = os.path.join(baseroot, sub_dir, 'inten/')
            sroot = os.path.join(baseroot, sub_dir, 'pha/')

            c_files = sorted(os.listdir(croot))
            s_files = sorted(os.listdir(sroot))

            c_image_paths.extend([os.path.join(croot, filename) for filename in c_files])
            s_image_paths.extend([os.path.join(sroot, filename) for filename in s_files])
        
        length = min(len(c_image_paths), len(s_image_paths))

        c_image_paths = c_image_paths[:length]
        s_image_paths = s_image_paths[:length]

        print(f"Total content paths: {len(c_image_paths)}")
        print(f"Total style paths: {len(s_image_paths)}")

        instance_paths = [] 
        return c_image_paths, s_image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True

