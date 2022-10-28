import pandas as pd
import numpy as np 

def make_dataset(study, base_path='/scratch/kmvr819/data/cbioportal/datahub/public', **kwargs):
    sample = pd.read_csv('{}/{}//data_clinical_sample.txt'.format(base_path, study), sep='\t', comment='#')
    patient = pd.read_csv('{}/{}/data_clinical_patient.txt'.format(base_path, study), sep='\t', comment='#')

    dataset = pd.merge(sample, patient, on='PATIENT_ID')
    
    modalities = []
    if kwargs.get('mutations', False) == True:
        mutations = pd.read_csv('{}/{}/data_mutations_extended.txt'.format(base_path, study), sep='\t', low_memory=False, comment='#')
    
        mutations = mutations[mutations['Variant_Classification'].isin([
            'Missense_Mutation', 
            'Frame_Shift_Del', 
            'Frame_Shift_Ins', 
            'Nonsense_Mutation'
        ])].reset_index(drop=True)

        mutations['Counter'] = 1

        mutations['Hugo_Symbol'] = 'molecular_' + mutations.Hugo_Symbol
        mutations = mutations[['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Counter']].groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).sum()[['Counter']].reset_index().pivot_table(index='Tumor_Sample_Barcode', columns='Hugo_Symbol', values='Counter').reset_index()
        mutations['study'] = study

        modalities.append(mutations)

    if kwargs.get('cnv', False) == True:
        cna = pd.read_csv('{}/{}/data_CNA.txt'.format(base_path, study), sep='\t', low_memory=False, comment='#')
        cna = cna.T
        cna.columns = "cnv_"+cna.iloc[0, :]
        cna = cna.iloc[1:, :]
        cna['SAMPLE_ID'] = cna.index
        cna = cna.reset_index(drop=True).replace(0, np.nan)

        modalities.append(cna)
    
    for mx, mod in enumerate(modalities):
        if mx == 0:
            dataset = pd.merge(dataset, mod, left_on='SAMPLE_ID', right_on='Tumor_Sample_Barcode', how='left').reset_index(drop=True)
        else:
            dataset = pd.merge(dataset, mod, on='SAMPLE_ID', how='left').reset_index(drop=True)
    return dataset
