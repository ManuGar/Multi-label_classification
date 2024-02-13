import pandas as pd
import numpy as np
import os
base = pd.read_excel("actionRename.xlsx", index_col=0)

baseDF = pd.DataFrame(base)

# print(baseDF)

cls = np.arange(13)
# print(base.shape)
for i,line in baseDF.iterrows():
  totFrames = line[35]
  # print(totFrames)
  ann = np.zeros((int(totFrames),3))
  for j in range(int(totFrames)):
    ann[j,0]=j
  ann[int(line[1]):int(line[2])+1,1]= cls[1]

  ann[int(line[3]):int(line[8])+1,1]= cls[2]
  ann[int(line[5]):int(line[6])+1,2]= 1
  ann[int(line[7]):int(line[8])+1,2]= 2

  ann[int(line[9]):int(line[14])+1,1]= cls[3]
  ann[int(line[11]):int(line[12])+1,2]= 1
  ann[int(line[13]):int(line[14])+1,2]= 2

  ann[int(line[15]):int(line[16])+1,1]= cls[4]
  ann[int(line[17]):int(line[18])+1,1]= cls[5]
  ann[int(line[19]):int(line[20])+1,1]= cls[6]
  ann[int(line[21]):int(line[22])+1,1]= cls[7]
  ann[int(line[23]):int(line[24])+1,1]= cls[8]
  ann[int(line[25]):int(line[26])+1,1]= cls[9]
  ann[int(line[27]):int(line[28])+1,1]= cls[10]

  ann[int(line[29]):int(line[32])+1,1]= cls[11]
  ann[int(line[31]):int(line[32])+1,2]= 1

  ann[int(line[33]):int(line[34])+1,1]= cls[12]

  if not os.path.exists('pruebas/annot_renamed'):
    os.mkdir('pruebas/annot_renamed')


  a_file = open('annot_renamed/'+line[0] + '.txt', "w")
  np.savetxt(a_file, ann, fmt="%d")
  a_file.close()


  # print(ann[line[34]-5:])
  # print(ann[:10])