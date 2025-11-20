<h3 align="center">NUMPY syntax cheatsheet</h3>

  <p align="center">
    Usefull functions
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>

## Table of contents

- [Simple array manipulation](#simple-array-manipulation)
- [Operations used for machine learning](#operations-used-for-machine-learning)

## Simple array manipulation
- **np.hstack** (sequence_of_ndarrays, dtype=None, casting=same_kind)
   - horizontally add two arrays to eachother

    In:
    ```python
    np.array([[1],[2],[3]]) 
    b = np.array([[4],[5],[6]])
    np.hstack((a,b)) 
    ```

    Out:

    ```python
    array([[1, 4],
        [2, 5],
        [3, 6]]) 
    ```

    | Visualisation|
    | -------- |
    ![hstack](/resources/imgs/hstack_example.png)

- np.**vstack**(*sequence_of_ndarrays, dtype=None, casting=same_kind*)
   - vertically add two arrays to eachother
   In:
    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    np.vstack((a,b))
    ```
    Out:
    ```python
    array([[1, 2, 3],
        [4, 5, 6]])
    ```
    | Visualisation|
    | -------- |
    ![hstack](/resources/imgs/vstack_example.png)

## Operations used for machine learning
- **One hot encoding** =  method for converting categorical variables into a binary format. It creates new columns for each category where 1 means the category is present and 0 means it is not
    In:
    ```python
    y = np.array([0, 1, 2, 3])
    y = to_categorical(y, 4) # converts a class vector (integers) to binary class matrix
    y
    ```
    Out:
    ```python
    array([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
    ```  
   - **Example**:
        - Imagine we have a **dataset with fruits** their categorical values and corresponding prices. Using one-hot encoding we can transform these categorical values into numerical form.
        - Wherever the fruit is "Apple," the **Apple column will have a value of 1** while the **other fruit** columns (like Mango or Orange) will contain **0**.
        - This pattern ensures that each categorical value gets its own column represented with binary values **(1 or 0)** making it usable for machine learning models.
![one-hot-encoding](/resources/imgs/NP_1.png)


