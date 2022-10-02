# Cellular uptake of random elliptic particles

## Associated thesis
The present code is the supplemental material associated to the [thesis](https://www.theses.fr/s234530).

## Dependencies
In order to make sure that you are able to run the code, please install the required versions of the libraries by executing the command bellow in your terminal.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```pip3 install -r requirements.txt```

## Tutorial
The repository is composed of a folder "uptake" itself composed of several folders:
- model: gathers the scripts to define the properties of the NP-membrane system, to compute the total potential energy of the system and to deduce the final wrapping phase. These information are exported in the format of a .pkl file.
- multiprocessing: contains the routines to define the cases to be studied, to compute them in
  parallel and to export the corresponding .pkl files.
- model_posttreatment: contains the scripts to extract the phases from the aforementioned .pkl files.
- metamodel_implementation: gathers the scripts for the creation and validation of metamodels.
- sensitivity_analysis: contains the routines to compute the Sobol sensitivity indices along with the study of their convergence.

The repository also contains a setup.py file, that should be installed with the following command:

```python -m pip install -e . --user```

To remove this setup from your machine, use the following command:

```pip uninstall np_uptake```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
A [GPL](https://tldrlegal.com/license/bsd-3-clause-license-(revised)) license is associated to this code, in the textfile license.txt.


