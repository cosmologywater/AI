import os, keras, struct
import numpy as np

##################################3 ##################################3 ##################################3 ##################################3
##################################3 ##################################3 ##################################3 ##################################3
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
##################################3 ##################################3 ##################################3 ##################################3
##################################3 ##################################3 ##################################3 ##################################3

models = os.popen('ls sgd_lr0.01_momentum0.9_denseplus/*.save').read().split()
print(models)

samples = []
for nowstr in ['BigMD', 'om_As_test']:
    samples = samples + os.popen('ls /home/xiaodongli/data/colas/cola_multiverse/'+nowstr+'/*grid*c.b').read().split()
print('In total ', len(samples), 'samples.')

def read_testdict(filename):
    nowdict = {}
    for nowstr in open(filename, 'r').readlines():
        nowstrs = nowstr.split()
        nowdict[nowstrs[0]] = nowstr
    return nowdict

def write_testdict(filename, nowdict):
    nowf = open(filename, 'w')
    for key in nowdict.keys():
        rlts = nowdict[key]
        for rlt in rlts.split():
            nowf.write(str(rlt)+' ')
        nowf.write('\n')

for model in list(models):
  try:
        print('\nload in model:', model)
        nowmodel = keras.models.load_model(model)
        outputfile = model + '.test_output'
        if os.path.isfile(outputfile):
            print(outputfile, 'found.')
            nowdict = read_testdict(outputfile)
        else:
            nowdict = {}
        allkeys = nowdict.keys()
        for sample in samples:
            if sample in allkeys:
                continue
            else:
                print(' * predict ', sample, '...')
                #y = []
                #for subcube in subcubes(load_grid(sample)):
                #    y.append(nowmodel.predict(subcube.reshape(-1,32,32,32,1)))
                #    print('predicted result: ', y)
                try:
                 y = nowmodel.predict(np.array(subcubes(load_grid(sample))).reshape(-1,32,32,32,1))
                 nowstr = sample+' '
                 for yy in y:
                    nowstr += (str(yy[0])+' '+str(yy[1])+'  ')
                 nowdict[sample] = nowstr+' '
                except:
              	  print(' ** failed to predict for', sample, '...')
        print('write to : ', outputfile)
        write_testdict(outputfile, nowdict)
  except: 
        pass






