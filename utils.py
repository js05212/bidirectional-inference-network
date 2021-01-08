import torch
import os
import zipfile
import glob
import shutil


def copy_key_src(dir_from, dir_to):
    os.makedirs(dir_to, mode=0o770)
    l_f = os.listdir(dir_from)
    for ele in l_f:
        if '.py' in ele:
            ele_to = ele.split('/')[-1]
            shutil.copyfile(dir_from + '/' + ele, dir_to + '/' + ele_to)

def save_model(model_names, models, output_folder, epoch, metric, info):
    model_path = '{}/{}_best_model_epoch{}_{:.4f}.zip'.format(output_folder, info, epoch, metric)
    for model, model_name in zip(models, model_names):
        with open('{}/{}_model_{}.pth'.format(output_folder, info, model_name), 'wb') as f:
            torch.save(model.state_dict(), f)
    for filename in glob.glob(os.path.join(output_folder, '{}_best_model*.zip'.format(info))):
        os.remove(filename)
    with zipfile.ZipFile(model_path, 'w') as f:
        for model_name in model_names:
            f.write('{}/{}_model_{}.pth'.format(output_folder, info, model_name), model_name)
            os.remove('{}/{}_model_{}.pth'.format(output_folder, info, model_name))

def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")