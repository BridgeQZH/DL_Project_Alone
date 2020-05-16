
import pickle
import numpy as np
import matplotlib.pyplot as plt


'''
To run this code, you need to download the data%d.p file which contains the accuracy.
And then change the path that fits your setting.
temp = {}
temp['epoch'] = epoch  # existing key, so overwrite
temp['n_epochs'] = self.n_epochs  # new key, so add
temp['i'] = i
temp['batch_size'] = self.batch_size
temp['D_loss']=loss_d[0]
temp['acc'] = 100*loss_d[1]
temp['G_loss'] = float("{:.5f}".format(loss_g[0]))
temp['adv'] = float("{:.5f}".format(np.mean(loss_g[1:3])))
temp['recon'] = float("{:.5f}".format(np.mean(loss_g[3:5])))
temp['id'] = float("{:.5f}".format(np.mean(loss_g[5:7])))
'''
avg_list = []
loss_list = []
for j in range(90): # This number depends the epoch we set at first
    name='epoch_'+str(j)
    locals()[name] = pickle.load( open("/home/zihan/Downloads/gan_data/monet2photo/data%d.p" %j, "rb" ))
    # 'avg_acc_epoch_'+str(i)
    # print(sum(locals()[name][i]['acc'] for i in range(10)))
    temp_sum = 0
    for i in range(len(locals()[name])):
        # temp_sum += locals()[name][i]['acc']
        temp_sum += locals()[name][i]['D_loss']
        # print(temp_sum)
    # print(temp_sum/len(locals()[name]))
    avg_list.append(temp_sum/len(locals()[name]))
    # print(avg_list)
    # print(sum(locals()[name][i]['acc'] for i in range(len(locals()[name]))) / len(locals()[name]))
# for i in range(len(avg_list)):
#     print(avg_list[i])
fig = plt.figure()
plt.plot(avg_list)
fig.suptitle('Monet2photo no flipped')
plt.xlabel('epoch')
plt.ylabel('average D_loss')
plt.show()

'''
To see at which epoch, we have the highest accuracy
'''
m = min(avg_list)
print(m)
print([i for i, j in enumerate(avg_list) if j == m])