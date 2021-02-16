import numpy as np
import matplotlib.pyplot as plt 
import argparse



start_plot_idx = 1
def parse_args():
    parser = argparse.ArgumentParser(description='show loss acc')
    parser.add_argument('--name', dest='name', help='Saved Pic Name', default='loss.png', type=str)

    args = parser.parse_args()
    return args


def show_loss(lossfile,iter_interval,statistic_interval,lineset,save_name):
    loss_file = open(lossfile, 'r')
    loss_total = loss_file.readlines()
    loss_num = len(loss_total)
    loss_res = np.zeros(loss_num)

    for idx in range(loss_num) :
        loss_str = loss_total[idx]
        str_start_idx = loss_str.find('mAP: ')+4
        str_end_idx = len(loss_str)-1
        tmp = loss_str[str_start_idx+1:str_end_idx]
        loss_res[idx] = float(tmp.strip('%'))

    print("max index {} & value {:.2f}".format(np.argmax(loss_res),np.max(loss_res)))
    save_name = save_name.split('.png')[0]+'_maxind{}_mAP{:.2f}.png'.format((np.argmax(loss_res)+1)*2,np.max(loss_res))
    
    statistic_len = (loss_num + statistic_interval-1)//statistic_interval
    statistic_idx = np.arange(statistic_len) * iter_interval * statistic_interval
    statistic_res_mean = np.zeros(statistic_len)
    statistic_res_var = np.zeros(statistic_len)

    print(statistic_idx)
    import pdb;pdb.set_trace()
    for idx in range(statistic_len) :
        loss_start_idx = idx*statistic_interval
        loss_end_idx = min(loss_start_idx + statistic_interval, loss_num)
        loss_part = loss_res[loss_start_idx : loss_end_idx]
        statistic_res_mean[idx] = np.mean(loss_part)
        statistic_res_var[idx] = np.var(loss_part)
        
        # print("start idx, ",loss_start_idx)
        # print("end idx, ",loss_end_idx)
        # print("loss_part: ",loss_part)
        # print("mean loss_part: {:.2f}".format(statistic_res_mean[idx]))
        # print("var loss_part: {:.2f}".format(statistic_res_var[idx]))
        # import pdb;pdb.set_trace()

    plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:], lineset)
    plt.title('mAP')
    plt.savefig(save_name)
    plt.show()
    print("save to ",save_name)

# python show_loss.py --name ../log/regdb_log/RGA_att34.png
args = parse_args()
save_name = args.name

# iter_interval: test epoch interval in *.log -> 横坐标
# statistic_interval : draw mean value of period -> 点的密集程度
show_loss('trainloss.refine',2,4,'r-o',save_name)


