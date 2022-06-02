from scipy import stats
from scipy.stats import skewtest, kurtosistest
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')


######################################################################
# Ben Terry 
# Data Science & Analytics
######################################################################

###### VARIABLES ######
alpha = .05

################ PART A ################
print("##################################")
print('Output from application TerryB_a3.py')
print('Developed by Ben Terry')
print("##################################\n")



# 1. Read the data file and assign Series object; find row count
print('\nInitiating scrubbing for part A...')
datfile='C:\\Users\\ben_i\\Documents\\Motor_Vehicle_Crashes_-_Individual_Information__Three_Year_Window.csv'

frame = pd.read_csv(datfile)

count = len(frame) # ROW COUNT


# 2. Drop unwanted columns
frame.drop(columns = ['Year', 'Case Individual ID', 'Case Vehicle ID', 'Victim Status', 'Role Type',
                      'Ejection', 'License State Code', 'Transported By', 'Safety Equipment',
                      'Injury Descriptor', 'Age', 'Injury Location'], axis = 1, inplace = True)


# 3. Select Driver-only entries from the ‘frame’ DataFrame and retain the result in ‘frame’
frame = frame[frame['Seating Position'] == 'Driver']


# 4. Rename the ‘Sex’ column as ‘Gender’.
frame.rename(columns = {'Sex' : 'Gender'}, inplace = True)


# 5. Develop logic that discards values not equal to 'M' or 'F'
genderList = frame['Gender'].tolist()

for i in range (0, len(genderList)):
    if genderList[i] != 'M' and genderList[i] != 'F':
        genderList[i] = None


sr = pd.Series(data = genderList)
frame['Gender'] = sr # Update scrubbed column


# 6. More Filtering
injuryList = frame['Injury Severity'].tolist()

for i in range (0, len(injuryList)):
    if injuryList[i] != 'Uninjured' and injuryList[i] != 'Minor' and injuryList[i] != 'Moderate' and injuryList[i] != 'Severe':
        injuryList[i] = None
    

sr2 = pd.Series(data = injuryList)
frame['Injury Severity'] = sr2 # Update scrubbed column


# 7. Final for part A
count2 = len(frame)
print('\n######## Scrubbing has been completed! ########')
print('\nOriginal number of data:', count)
print('Final size of data (after scrubbing):', count2)


################ PART B ################
print('\n\nInitiating part B...')
print('\n\nNonparametric techniques will be used moving forward...')


################ PART C ################
print('\n\nInitiating part C...')


# 1. Binomial Test
print('\n\nNumber of males and females:\n')

mCount = len(frame[frame['Gender'] == 'M'])
fCount = len(frame[frame['Gender'] == 'F'])

print('Male:', mCount)
print('Female:', fCount, '\n')
print('P-value of binomial test: p =', stats.binom_test(mCount, n = count2, p = alpha))
print('\nResults show significant difference between the expected and observed gender frequencies.')


# 2. Chi-Squared Test
print('\n\nPerforming chisquared test...')

# Male counts
mUninjured = len(frame[(frame['Gender'] == 'M') & (frame['Injury Severity'] == 'Uninjured')])
mMinor = len(frame[(frame['Gender'] == 'M') & (frame['Injury Severity'] == 'Minor')])
mModerate = len(frame[(frame['Gender'] == 'M') & (frame['Injury Severity'] == 'Moderate')])
mSevere = len(frame[(frame['Gender'] == 'M') & (frame['Injury Severity'] == 'Severe')])

# Female counts
fUninjured = len(frame[(frame['Gender'] == 'F') & (frame['Injury Severity'] == 'Uninjured')])
fMinor = len(frame[(frame['Gender'] == 'F') & (frame['Injury Severity'] == 'Minor')])
fModerate = len(frame[(frame['Gender'] == 'F') & (frame['Injury Severity'] == 'Moderate')])
fSevere = len(frame[(frame['Gender'] == 'F') & (frame['Injury Severity'] == 'Severe')])

# Display counts
print('\n\nMale injury counts by severity [Uninjured, Minor, Moderate, Severe]:\n[', 
      mUninjured, ',', mMinor, ',', mModerate, ',', mSevere, ']')
print('\nFemale injury counts by severity [Uninjured, Minor, Moderate, Severe]:\n[', 
      fUninjured, ',', fMinor, ',', fModerate, ',', fSevere, ']\n')

# scipy.stats.chi2_contingency() to perform a chisquared test of independence
chiSqr = stats.chi2_contingency([[mUninjured, mMinor, mModerate, mSevere], 
                                [fUninjured, fMinor, fModerate, fSevere]])


print('\nChi-squared results:\n')
print('Test statistic:', format(chiSqr[0], '.2f'), '\nP-value =', chiSqr[1])
print('\nResults display that the injury severities between male and female'\
      ' are significantly different.\n\n')

print('################ END OF PROGRAM ################')















