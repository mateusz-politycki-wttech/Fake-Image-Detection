from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/test/'

# # list of synthesis algorithms
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal',
        'dalle_2']

# vals = ['dalle_2']

# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# multiclass = [0]

# model
# model_path = 'weights/blur_jpg_prob0.5.pth'
model_path = 'checkpoints/blur_jpg_prob0.5/model_epoch_best.pth'
