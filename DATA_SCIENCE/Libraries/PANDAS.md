<h3 align="center">PANDAS syntax cheatsheet</h3>

  <p align="center">
    Usefull functions
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>

## Table of contents
- [Working on dataframes](#working-on-dataframes)



## Working on dataframes
- **reset_index**(*drop=*) 
    - resets index in dataframe, 
    - **drop=True** removes unnecessary column with previous indexing
   ```python
   df = df.reset_index(drop=True)
   ```

- **.unique()** 
   - get unique values from dataframe / dataframe's column
   ```python
   unique_values = df['col_name'].unique()
   ```
- **.value_counts**(*subset=, normalize=, sort=, ascending=, dropna=*)
   - returns a series containing the frequency of each unique row in the dataframe
   In:
   ```python
   df = pd.DataFrame({'col1': [2, 4, 4, 6],
                        'col2': [2, 0, 0, 0]})
   unique_rows = df.value_counts()

   print(df)
   print(unique_rows)
   ```
   Out:
   ```python
   # index cols
         col1  col2
   0     2     2 # <- unique row
   1     4     0 # <- duplicate row
   2     4     0 # <- duplicate row
   3     6     0 # <- unique row
   # cols      count
   col1  col2
   4     0       2 
   2     2       1
   6     0       1
   ```

- **.concat()**
   - concats multiple dataframes into one (they have to have the same shape and cols)
   - ingore index = just adds them after eachother based on order in list
   ```python
   df_combined = pd.concat([df1, df2], ignore_index=True) 
   ```
- **.concat()** dataframes with columns with **different lengths**:
   - the shorter column will get NaN values where longer column still has values
   In:
   ```python
   df_long = pd.DataFrame({
               'Column_long':[1, 2, 3, 4, 5, 6] })

   df_short = pd.DataFrame({
         'Column_short': [1,2,3] })

   df_combined = pd.concat([df_long, df_short], axis=1) 

   print(df_combined)
   ```
   Out:
   ```python
               Column_long  Column_short
   0            1           1.0
   1            2           2.0
   2            3           3.0
   3            4           NaN
   4            5           NaN
   5            6           NaN
   ```

- **.insert()**
   - inserts a new column to a dataframe at a certain location
   In:
   ```python
   df = pd.DataFrame({'col2': [2, 4, 4, 6, 7],
                           'col3': [2, 0, 0, 0, 2]})
   new_col = [1, 2, 3, 4, 5]
   df.insert(loc = 0,     # location
         column = 'col1', # name 
         value = new_col)
   ```
   Out:
   ```python
         col1  col2  col3
   0     1     2     2
   1     2     4     0
   2     3     4     0
   3     4     6     0
   4     5     7     2
   ```

- using **.loc()** to select **rows** that contain a certain value / matches certain **conditions** in a specified column
   ```python
   selected_rows1 = df.loc[df['column_name'] == some_value] 
   # conditions
   selected_rows2 = df.loc[(df['column_name'] >= condition_1) 
   & (df['column_name'] <= condition_2)]

   # get the count of these rows:
   count_conditioned_rows = (df['column_name'] == some_value).sum()
   ```

- create dataframe from lists:
   In:
   ```python
   list_of_lists = [['tom', 25], ['krish', 30],
                     ['nick', 26], ['juli', 22]]

   df = pd.DataFrame(list_of_lists, columns = ['Name', 'Age'])
   print(df)
   ```

   Out:
   ```python
         Name  Age
   0    tom   25
   1  krish   30
   2   nick   26
   3   juli   22
   ```

- get the **first row** of dataframe **where [condition/s]**
   ```python
   first_row = df[df['col_name'] == value].iloc[0]
   first_row = df[df['col_name'] > value].iloc[0]
   first_row = df[(df['col_name'] > value1) 
   & (df['col_name'] < value2)].iloc[0]

   # get index of that first row
   first_row_index = df[df['col_name'] == value].index[0]
   ```
- **.iloc()** 
   - slicing dataframe's rows / columns based on indexes
   In:
   ```python

   list_of_lists = [['tom', 25, 160, 'No'], ['krish', 30, 170, 'No'],
         ['nick', 26, 180, 'Yes'], ['juli', 22, 190, 'Yes']]

   df = pd.DataFrame(list_of_lists, columns = ['Name', 'Age', 'Heigh', 
                                          'Yes?'])
   # keep all COLUMNS and only first 2 ROWS (0-1)
   df_rows = df.iloc[0:2]
   # keep all ROWS and only first 2 COLUMNS (0-1)
   df_cols = df.iloc[:, 0:2]

   print(df)
   print(df_rows)
   print(df_cols)
   ```
   Out:
   ```python
   Name  Age  Heigh Yes?
   0    tom   25    160   No
   1  krish   30    170   No
   2   nick   26    180  Yes
   3   juli   22    190  Yes
   --------------------          # ALL cols, 2 rows
   Name  Age  Heigh Yes?         
   0    tom   25    160   No
   1  krish   30    170   No
   --------------------          # 2 cols, ALL rows
   Name  Age                     
   0    tom   25
   1  krish   30
   2   nick   26
   3   juli   22
   ```

- **melt** columns into one
   In:
   ```python
   A    B    C D
   Cat Meow  1 2
   Dog Grr   3 4
   ```
   ```python
   df.melt(['C', 'D'], var_name='Level', value_name='Focus')
   ```
   Out:
   ```python
      C  D Level Focus
   0  1  2     A   Cat
   1  3  4     A   Dog
   2  1  2     B  Meow
   3  3  4     B   Grr
   ```

- converting columns into **series**:
   ```python
   series_1= df.ix[:,0] # 0 - first column
   ```
