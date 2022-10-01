import pandas as pd
import numpy as np 

def mutations2matrix(study, base_path='/scratch/kmvr819/data/cbioportal/datahub/public'):
    sample = pd.read_csv('{}/{}//data_clinical_sample.txt'.format(base_path, study), sep='\t', comment='#')
    patient = pd.read_csv('{}/{}/data_clinical_patient.txt'.format(base_path, study), sep='\t', comment='#')

    clinical = pd.merge(sample, patient, on='PATIENT_ID')

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

    dataset = pd.merge(clinical, mutations, left_on='SAMPLE_ID', right_on='Tumor_Sample_Barcode', how='left').reset_index(drop=True)
    
    return dataset