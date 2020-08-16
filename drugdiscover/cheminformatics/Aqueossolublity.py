
#dataset
#! wget https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt

import pandas as pd

delaney_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv'
sol = pd.read_csv(delaney_url)
sol

from rdkit import Chem

mol_list = [Chem.MolFromSmiles(element) for element in sol.SMILES]

"""3.2. Calculate molecular descriptors
To predict LogS (log of the aqueous solubility), the study by Delaney makes use of 4 molecular descriptors:

cLogP (Octanol-water partition coefficient)
MW (Molecular weight)
RB (Number of rotatable bonds)
AP (Aromatic proportion = number of aromatic atoms / total number of heavy atoms)
fortunately, rdkit readily computes the first 3. As for the AP descriptor, we will calculate this by manually computing the ratio of the number of aromatic atoms to the total number of heavy atoms which rdkit can compute.
"""

import numpy as np
from rdkit.Chem import Descriptors

#first 3 descriptors


def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)
           
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i == True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR

X = generate(sol.SMILES)

#Y matrix
Y = sol.iloc[:,1]
Y = Y.rename("logs")
Y

dataset = pd.concat([X,Y], axis=1)
dataset.to_csv('delaney_solubility_with_descriptors.csv', index=False)

#machine learning model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#check prediction for training data
Y_pred_train = model.predict(X_train)

print('Coefficient',model.coef_)
print('Intercept', model.intercept_)

# check predictions on test dataset
Y_pred_test = model.predict(X_test)
print('Coefficient',model.coef_)
print('Intercept', model.intercept_)

#logs
yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
Mw = '%.4f MW' % model.coef_[1]
Rb = '%.4f RB' % model.coef_[2]
Ap = '%.2f' % model.coef_[3]

