# Work in progress control methods

Control methods implemented in C++ with design/visualization in Python. 

System matrix
```
    A =  [[ 0.          0.          0.        ]
         [-0.19740973  0.19740973  0.        ]
         [ 0.         -2.012       0.        ]]

    B = [[-0.35052265]
        [-0.00997366]
        [ 0.        ]]
```

states
```
\psi_1 // Truck orientation
\psi_2 // Trailer orientation
y_2 // lateral position of trailer axle
```

## Setup 

- Python 3.11
- C++ (I'm using vscode with msvs 2022)
- [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page)

### Python

Install [miniconda](https://docs.anaconda.com/miniconda/)

```
conda create -n envname python=3.11
```

Then install the following libraries

```
pip install -r requirements.txt
```

### C++

Many choices here, but it is noteworthy that the Eigen library requires you to include the path to the library during compile time

```
-I <PATH>/eigen-3.4.0
```

## Usage

Get the feedback control gains printed out
```
python design.py
```

Compile and run C++ and two files will be generated to csv. 


Takes the two csv files and plots the results
```
python display.py
```