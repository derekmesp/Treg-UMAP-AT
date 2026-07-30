import pandas as pd

donors = pd.read_excel(input('Enter the path to the donor data file:'))
CMV = donors[donors['Ab CMV:'] == 'Positive']
CMV = CMV[CMV['HLA-DR'].astype(str).str.contains('15')]
EBV = donors[donors['Ab EBV (IgG):'] == 'Positive']
EBV = EBV[EBV['HLA-DR'].astype(str).str.contains('4')]
HLADR = donors[donors['HLA-DR'].astype(str).str.contains('4')]

CMV.to_excel('CMV Filtered.xlsx')
EBV.to_excel('EBV Filtered.xlsx')
HLADR.to_excel('HLA-DR Filtered.xlsx')
