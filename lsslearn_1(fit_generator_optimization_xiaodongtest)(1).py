#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('pylab', 'inline')

A = np.random.rand(3,3, )
print (A)

def data_aug2d(A):
    return [A, np.fliplr(A), np.rot90(A), np.fliplr(np.rot90(A)), rot90(rot90(A)), np.fliplr(rot90(rot90(A))), 
     rot90(rot90(rot90(A))), np.fliplr(rot90(rot90(rot90(A))))]

fig, axs = subplots(3,3, figsize=(9,9)); row = 0
As = data_aug2d(A)
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(As[row]); row+=1
        if row == 8: break
fig.tight_layout()


# ### 以上是数据增强的测试代码，还没成功，忽略。。。从这里开始

# ##### for some cosmologies data are missing...!!!??? we have to skip them

# In[6]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import keras, os, pynbody, struct

lsstr = "ls /media/xiaodongli/0B9ADFB4341AD2BD/om_As/"

def cosmostr(om, As):
    return 'om%.3f' % om + '_As%.3f' % As

def snpfiles(cosmology, snpstr='c'):  #
    return os.popen(lsstr + cosmology + "*snap*" + snpstr + ".*").read().split()

def gridfiles(cosmology, snpstr='c'):
    return os.popen(lsstr + cosmology + "*grid*" + snpstr + ".*").read().split()

def mocklist():
    files = os.popen(lsstr + "om*.lua").read().split('\n')
    # *代替多个字母,即列出所有符合条件的.lua文件:om...
    cosmologies = []  # 宇宙学参数
    mocks = {}  # 模拟测试
    ifile = 0  # 有效文件
    for nowfile in files:
        # str[a:b]不存在时,返回'',不存在则忽略
        nowstr = nowfile[-39:-10]
        if nowstr == '':
            continue
        cosmologies.append(nowstr[0:15])
        ifile += 1
        try:
            mocks[nowstr[0:15]] = {'om': float(nowstr[2:7]), 'As': float(nowstr[10:15]),
                                   'sigma8': float(nowstr[23:29])}
            # 添加随机数种子
            ranseed = float(open(nowfile, 'r').readline().split()[2])  # 默认以所有空字符为分隔符,包括空格,\n,\t
            mocks[nowstr[0:15]]['ranseed'] = int(ranseed)
            # print(ranseed)
        except:
            pass
    return cosmologies, files, mocks

gridfile_dict = {}

cosmologies, filenames, infos = mocklist()
print('In total ', len(cosmologies), 'cosmologies')

print('Build up gridfile_dict... (for speed-up of load_grid()) ')
for cosmology in cosmologies:
    rlt = gridfiles(cosmology)
    if rlt == []:
        print ('\tmissing cosmology!', cosmology)
    else:
        gridfile_dict[cosmology] = rlt[0]
np.random.shuffle(cosmologies)


# In[7]:


def load_grid(gridfile, snpstr='c', printinfo=False):  # 网格加载
    #gridfile = os.popen(lsstr + cosmology+"_sigma8_*grid*" + snpstr + ".*").read().split()[0]
    #print('load in gridfile : ', gridfile, '...')
    nowf = open(gridfile, 'rb')  # 以二进制形式读取文件
    # struct:对python基本类型值与用python字符串格式表示的C struct类型间转化
    size = struct.unpack('f' * 1, nowf.read(4 * 1))[0]
    grid_nc = struct.unpack('i' * 1, nowf.read(4 * 1))[0]
    data = struct.unpack('f' * grid_nc ** 3, nowf.read(4 * grid_nc ** 3))
    if printinfo:
        print('read in box size     \n\t', size)
        print('read in num_grid      \n\t', grid_nc)
        print('read in coarse grid \n\tsize    : ', len(data), '\n\texpect  : ', grid_nc ** 3)

    nowf.close()
    return np.array(data).reshape((grid_nc, grid_nc, grid_nc))

def subcubes(A):
    rlt = []
    for row1 in [0, 32, 64, 96]:
        for row2 in [0, 32, 64, 96]:
            for row3 in [0, 32, 64, 96]:
                rlt.append(A[row1:row1+32,row2:row2+32,row3:row3+32])
    return rlt

def data_augument(A):
    rlt = []


# In[8]:


n_train = int(len(cosmologies)*4/5)
print(n_train)
cosmologies_train, cosmologies_test = cosmologies[0:n_train], cosmologies[n_train:]


# In[17]:


A = load_grid(gridfile_dict[cosmologies[0]])


# In[18]:


A.shape


# In[9]:


# lsstr = "ls /media/minstrel/Seagate/cola_multiverse/om_As/"
# 初始化数据集,因函数定义中参数不可转为全局变量
# 此过程受train_test_split的test_size参数的影响,则今后更改或维护程序应注意
test_size = 0.3
batch_size = int(n_train / 15)
num_subcube = 64
num_data_augument = 48

x_train = np.zeros((batch_size-int(test_size*batch_size)-1, 2))  # 因test_size=0.3,保证初始化矩阵的形状与后面的相同
x_test = np.zeros((int(test_size*batch_size)+1,32,32,32, 1))
y_test = np.zeros((int(test_size*batch_size)+1, 2))
y_train = np.zeros((batch_size-int(test_size*batch_size)-1, 2))


###  xiaodong: 重新写了 load_grid 程序。。。之前有错误！！！！（好像只会 load 进来一个 om...)
def train_generator():  # 必须无限循环yield数据,全部数据遍历后再重新遍历数据,为下一个epoch yield 数据
    i = 0
    while 1:
        X = []
        y = []
        global x_train, x_test, y_test, y_train, test_size, batch_size
        #print(' load in ', batch_size * i, 'to', batch_size * (i + 1), '... len(cosmologies)=',
        #      len(cosmologies))
        for cosmology in cosmologies[batch_size * i: batch_size * (i + 1)]:
            try:
                gridfile = gridfile_dict[cosmology]
                gridfile_exist = True
            except:
                #print('skip cosmology ', cosmology, '!!!')
                gridfile_exist = False
            if gridfile_exist:
                griddata = load_grid(gridfile, 'c')
                for subcube in subcubes(griddata):
                    X.append(subcube)
                    y.append(np.array([infos[cosmology]['om'], infos[cosmology]['sigma8']]))
        X = np.array(X)
        y = np.array(y)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)
        x_train = x_train.reshape(-1, 32, 32, 32, 1)
        x_test = x_test.reshape(-1, 32, 32, 32, 1)
        i += 1
        yield x_train, y_train  # tuple 类型
        # 15个批次后重新遍历数据,此循环即死循环
        if i == 465//batch_size:
            i = 0


# In[46]:


from keras import Sequential, layers
from sklearn import model_selection, metrics

model1 = keras.Sequential([
    layers.BatchNormalization( input_shape=(32, 32, 32, 1)),
    layers.Conv3D(32, (3, 3, 3), activation='relu'),
    layers.AveragePooling3D(pool_size=(2, 2, 2)),
    layers.BatchNormalization(),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.AveragePooling3D(pool_size=(2, 2, 2)),
    layers.BatchNormalization(),
    layers.Conv3D(128, (3, 3, 3), activation='relu'),
    layers.AveragePooling3D(pool_size=(2, 2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(2, ),
])
model1.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error',
              metrics=['mean_squared_error'])



# *************************************************************************************************
model1.fit_generator(train_generator(),
                    steps_per_epoch=465//batch_size,  # 数据规格可能大小不对应
                    epochs=100,
                    verbose=1,
                    validation_data=(x_test,y_test))


# In[ ]:


from keras import Sequential, layers
from sklearn import model_selection, metrics

model2 = keras.Sequential([
    layers.BatchNormalization( input_shape=(32, 32, 32, 1)),
    layers.Conv3D(2, (3, 3, 3), activation='relu', ),
    layers.BatchNormalization(),
    layers.Conv3D(12, (3, 3, 3), activation='relu', ),
    layers.AveragePooling3D(pool_size=(2, 2, 2),),
    layers.BatchNormalization(),
    layers.Conv3D(64, (9, 9, 9), activation='relu', ),
    layers.BatchNormalization(),
    layers.Conv3D(64, (3, 3, 3), activation='relu', ),
    layers.BatchNormalization(),
    layers.Conv3D(128, (2, 2, 2), activation='relu', ),
    layers.BatchNormalization(),
    layers.Conv3D(128, (2, 2, 2), activation='relu', ),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, ),
])
model2.compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error',
              metrics=['mean_squared_error'])



# *************************************************************************************************
model2.fit_generator(train_generator(),
                    steps_per_epoch=465//batch_size,  # 数据规格可能大小不对应
                    epochs=300,
                    verbose=1,
                    validation_data=(x_test,y_test))


# ### Test section

# In[10]:


def create_validate_sample(nsample, use_random=True, startid=None ):
    cosmologies = list(gridfile_dict); ncosmo = len(cosmologies)
    if use_random:
        rows = [np.random.randint(0,ncosmo) for row in range(nsample)];
        rows = list(set(rows))
        while len(rows) < nsample:
            rows = rows + [np.random.randint(0,ncosmo) for row in range(nsample - len(rows))];
            rows = list(set(rows))
    else:
        rows = range(startid, startid+nsample)
    x, y =[], []
    for row in rows:
        cosmology = cosmologies[row]
        gridfile = gridfile_dict[cosmology]
        griddata = load_grid(gridfile)
        for subcube in subcubes(griddata):
            x.append(subcube)
            y.append(np.array([infos[cosmology]['om'], infos[cosmology]['sigma8']]))
    x = np.array(x); x = x.reshape(-1, 32, 32, 32, 1); y = np.array(y)
    return x, y

def plot_test(model, x, y, plot_avg_predict = True, fig=None, ax = None, plot_subpoints=False):
    y_predict = model.predict(x); 
    
    if fig == None or ax ==None:
        fig, ax = subplots(figsize=(14,6))
    cs = range(len(y)); cs = cs / mean(cs)
    ax.scatter(y[:,0], y[:,1], c='b', marker='*', label='input', s=200)
    if plot_subpoints:
        ax.scatter(y_predict[:,0], y_predict[:,1], c='g',  marker='p', s=50, label='outputs')
    
        
    om_test, w_test = y[:,0], y[:,1]
    om_predict, w_predict = y_predict[:,0], y_predict[:,1]
    
    if plot_avg_predict:
        ax.scatter(mean(y_predict[:,0]), mean(y_predict[:,1]), marker='*', c='r', label='output_avg', s=200)
        ax.plot([om_test[0], mean(y_predict[:,0])], [w_test[0], mean(y_predict[:,1])], lw=2, c='k', ls='--' )

    
    #for row in range(len(om_predict)):
    #    ax.plot( [om_predict[row], om_test[row]], [w_predict[row], w_test[row]], lw=0.5, c='gray' )
    ax.set_xlabel(r'$\Omega_m$',fontsize=16); ax.set_ylabel(r'$\sigma_8$',fontsize=16)
    ax.legend()   
    return fig, ax


# In[56]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model1, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# ##### model1, 10 epochs: Don't run it! (参数已改不可恢复）

# In[176]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model1, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# ##### 50 epochs`5

# In[181]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model1, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# ##### 100 epochs

# In[52]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model1, x_test, y_test, fig=fig, ax=ax)
ax.set_title('#-epochs = '+str(100), fontsize=16)
ax.grid(); plt.show()


# ##### 207 epochs

# In[8]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model1, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# #### This shows how to save and load a model

# In[68]:



model1.save('model1.h5')

get_ipython().system('ls model1.h5 -alh')

model1_load = keras.models.load_model('model1.h5')

print(model1.predict(x_test[0:1]))
print(model1_load.predict(x_test[0:1]))


# In[69]:


models = []

imodel = 0

for epochs in [5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300, 500]:

    models.append(keras.Sequential([
        layers.BatchNormalization( input_shape=(32, 32, 32, 1)),
        layers.Conv3D(32, (3, 3, 3), activation='relu'),
        layers.AveragePooling3D(pool_size=(2, 2, 2)),
        layers.BatchNormalization(),
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.AveragePooling3D(pool_size=(2, 2, 2)),
        layers.BatchNormalization(),
        layers.Conv3D(128, (3, 3, 3), activation='relu'),
        layers.AveragePooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(2, ),
    ]))
    models[imodel].compile(optimizer=keras.optimizers.Adadelta(), loss='mean_squared_error',
              metrics=['mean_squared_error'])


    print('#########################################')
    print('Do a fit for ', epochs, 'epochs:')
    print('#########################################')
    # *************************************************************************************************
    models[imodel].fit_generator(train_generator(),
                    steps_per_epoch=465//batch_size,  # 数据规格可能大小不对应
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test,y_test))
    # Plot validation
    fig, ax = None, None
    for row in range(10):
        x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
        fig, ax = plot_test(models[imodel], x_test, y_test, fig=fig, ax=ax)
    ax.grid(); plt.show()
    ax.set_title('#-epochs = '+str(100), fontsize=16)
    imodel += 1


# In[73]:


row = 0
model_dict = {}
for epochs in [5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300, ]:
    model_dict[epochs] = models[row]; row+=1


# ##### save the models to files

# In[94]:


get_ipython().system('mkdir ./model1')
nowf = open('./model1/info.txt', 'w')
nowf.write('This is the first cnn model we use. See "lsslearn_1(fit_generator_optimization_xiaodongtest).ipynb" for details.')
nowf.close()

# Don't run this!! [may overwrite the files...]
if False:
    for key in model_dict.keys():
        model = model_dict[key]
        filepath = './model1/'+str(key)+'.save'
        keras.models.save_model(model, filepath)


# # Result and Test of model 1

# ##### load the models

# In[11]:


loaded_model = {}
for key in [5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300, ]:
    loaded_model[key] = keras.models.load_model('./model1/'+str(key)+'.save')


# ##### the input-output test

# In[13]:


for rand in range(5):
    fig, axs = plt.subplots(nrows=2, ncols = 6, figsize=(18,6))
    axs = axs.reshape(-1)
    for row in range(1):
        x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
        iax = 0; 
        for key in loaded_model.keys():
            #model = model_dict[key]    
            model = loaded_model[key]
            fig, ax = fig, axs[iax]
            #print(fig, ax)
            fig, ax = plot_test(model, x_test, y_test, fig=fig, ax=ax, plot_subpoints=True)
            ax.set_xlim(0.1, 0.6); ax.set_ylim(0., 1.2); ax.grid()
            ax.set_title('#-epochs = '+str(key),fontsize=16)
            iax += 1
    #for ax in axs:
    #    ax.set_xlim(0.1, 0.4); ax.set_ylim(0.3, 0.5)
    #    ax.set_title('#-epochs = '+str(key),fontsize=16)
    #    ax.grid(); 
    plt.show()   


# ##### model2, 10 epochs: Don't run it! (参数已改不可恢复）

# In[177]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model2, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# ##### 50 epochs

# In[182]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model2, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()


# ##### 300 epochs

# In[ ]:


fig, ax = None, None
for row in range(10):
    x_test, y_test = create_validate_sample(1, use_random=True, startid=row)
    fig, ax = plot_test(model2, x_test, y_test, fig=fig, ax=ax)
ax.grid(); plt.show()

