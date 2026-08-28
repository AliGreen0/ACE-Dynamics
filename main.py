from torch import true_divide
from train_test import train
import numpy as np

if __name__ == "__main__":    
    data_folder = 'LGG' # 'BRCA', 'ROSMAP', 'LGG', 'KIPAN'
    testonly = False
    modelpath = './model/'

    first_list = []
    second_list = []
    third_list = []

    for i in range (20):
        print(i+1, 'th run:')
        first, second, third, num_class = train(data_folder, modelpath, testonly)
        first_list.append(first)
        second_list.append(second)
        third_list.append(third)
    
    print('Final Results:')
    if num_class == 2:
        print('Accuracy: {:.5f} ± {:.5f}'.format(np.mean(first_list), np.std(first_list)))
        print('F1: {:.5f} ± {:.5f}'.format(np.mean(second_list), np.std(second_list)))
        print('AUC: {:.5f} ± {:.5f}'.format(np.mean(third_list), np.std(third_list)))
    else:
        print('Accuracy: {:.5f} ± {:.5f}'.format(np.mean(first_list), np.std(first_list)))
        print('F1 weighted: {:.5f} ± {:.5f}'.format(np.mean(second_list), np.std(second_list)))
        print('F1 macro: {:.5f} ± {:.5f}'.format(np.mean(third_list), np.std(third_list)))
