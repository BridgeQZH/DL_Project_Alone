
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
D_loss_list = []
G_loss_list = []
adv_loss_list = []
recon_loss_list = []
id_loss_list = []
# temp['adv'] = float("{:.5f}".format(np.mean(loss_g[1:3])))
#                 temp['recon'] = float("{:.5f}".format(np.mean(loss_g[3:5])))
#                 temp['id'] = float("{:.5f}".format(np.mean(loss_g[5:7])))
# for j in range(90): # For the monet2photo initial dataset

for j in range(68): # This number depends the epoch we set at first
    name='epoch_'+str(j)
    locals()[name] = pickle.load( open("/home/zihan/Downloads/gan_data/monet2photo_data_aug/nnnew_fliped_data%d.p" %j, "rb" ))
    # 'avg_acc_epoch_'+str(i)
    # print(sum(locals()[name][i]['acc'] for i in range(10)))
    temp_sum = 0
    temp_sum2 = 0
    temp_sum3 = 0
    temp_sum4 = 0
    temp_sum5 = 0
    for i in range(len(locals()[name])):
        # temp_sum += locals()[name][i]['acc']
        temp_sum += locals()[name][i]['D_loss']
        temp_sum2 += locals()[name][i]['G_loss']
        temp_sum3 += locals()[name][i]['adv']
        temp_sum4 += locals()[name][i]['recon']
        temp_sum5 += locals()[name][i]['D_loss']
        # temp_sum2 += locals()[name][i]['G_loss']
        # print(temp_sum)
    # print(temp_sum/len(locals()[name]))
    D_loss_list.append(temp_sum/len(locals()[name]))
    G_loss_list.append(temp_sum2/len(locals()[name])/20)
    adv_loss_list.append(temp_sum3/len(locals()[name]))
    recon_loss_list.append(temp_sum4/len(locals()[name]))
    id_loss_list.append(temp_sum5/len(locals()[name]))
    # G_loss_list.append(temp_sum2/len(locals()[name])/20)
    # print(avg_list)
    # print(sum(locals()[name][i]['acc'] for i in range(len(locals()[name]))) / len(locals()[name]))
# for i in range(len(avg_list)):
#     print(avg_list[i])
fig = plt.figure()
plt.plot(D_loss_list, color='red', label='Discriminator loss')
plt.plot(G_loss_list, color='green', label='Generator loss / 20')
plt.legend()
fig.suptitle('Discriminator and generator loss Monet2photo with data augmentation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

fig = plt.figure()
plt.plot(adv_loss_list, color='red', label='adversarial loss')
plt.plot(recon_loss_list, color='green', label='reconstruction loss')
plt.plot(id_loss_list, color='skyblue', label='identity loss')
plt.legend()
fig.suptitle('adversarial, reconstruction and identity loss Monet2photo with data augmentation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

'''
To see at which epoch, we have the highest accuracy
'''
# m = min(G_loss_list)
# print(m)
# print([i for i, j in enumerate(avg_list) if j == m])