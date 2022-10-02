import openturns as ot
import openturns.viewer as viewer
import time
ot.Log.Show(ot.Log.NONE)
import seaborn as sns

import numpy as np

import uptake.metamodel_implementation.utils as miu
from uptake.figures.utils import CreateFigure, Fonts, SaveFigure
from uptake.metamodel_implementation.metamodel_validation import MetamodelPostTreatment
from SALib.analyze import dgsm
import pickle

class Distribution:
    def __init__(self):
        """
        Constructs all the necessary attributes for the Distribution object.

        Parameters:
            ----------
            None

        Returns:
            -------
            None
        """

        self.gamma_bar_0_min = 1
        self.gamma_bar_0_max = 8
        self.sigma_bar_min = 0.5
        self.sigma_bar_max = 5.5
        self.gamma_bar_r_min = 1
        self.gamma_bar_r_max = 6
        self.gamma_bar_fs_min = -0.45
        self.gamma_bar_fs_max = 0.45
        self.gamma_bar_lambda_min = 10
        self.gamma_bar_lambda_max = 100

    def uniform(self):
        """
        creates a uniform distribution of the 3 input parameters

        Parameters:
            ----------
            None

        Returns:
            -------
            distribution: ot class
                uniform distribution of the 3 input parameters, computed wth Openturns.
                gamma_bar_lambda and sigma_bar_lambda could have been computed as constant values
                but we chose to generate them as uniform distribution with close bounds to match
                the architecture of the openturns library

        """
        distribution = ot.ComposedDistribution(
            [
                ot.Uniform(self.gamma_bar_0_min, self.gamma_bar_0_max),
                ot.Uniform(self.sigma_bar_min, self.sigma_bar_max),
                ot.Uniform(self.gamma_bar_r_min, self.gamma_bar_r_max),
                ot.Uniform(self.gamma_bar_fs_min, self.gamma_bar_fs_max),
                ot.Uniform(self.gamma_bar_lambda_min, self.gamma_bar_lambda_max),
            ]
        )
        return distribution


# Saltelli#
def compute_sensitivity_algo_Saltelli(distribution, metamodel, sensitivity_experiment_size):
    """
    computes the sensitivity algorithms computed after the Saltelli method

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        metamodel: ot class
            metamodel (Kriging) computed in the metamodel_creation.py script and stored in a .pkl
            file
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
        -------
        sensitivityAnalysis: ot class
            sensitivity algorithm computed with the openturns class

    """
    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Saltelli(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Saltelli",
):
    """
    computes the sensitivity algorithms computed after the Saltelli method and exports it to a .pkl
    file

    Parameters:
        ----------
        type_of_metamodel: str
            type of metamodel that has been computed. Possible value: "Kriging"
        training_amount: float
            proportion (between 0 and 1) of the initial data used for training (the remaining data
            are used for testing)
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity
        type_of_Sobol_sensitivity_implementation: str
            type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
            sensitivity algorithm is stored.

    Returns:
        -------
        None

    """
    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Saltelli = compute_sensitivity_algo_Saltelli(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(
        type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Saltelli, complete_pkl_filename_sensitivy_algo)


# Jansen#
def compute_sensitivity_algo_Jansen(distribution, metamodel, sensitivity_experiment_size):
    """
    computes the sensitivity algorithms computed after the Jansen method

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        metamodel: ot class
            metamodel (Kriging) computed in the metamodel_creation.py script and stored in a .pkl
            file
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
        -------
        sensitivityAnalysis: ot class
            sensitivity algorithm computed with the openturns class

    """
    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.JansenSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Jansen(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Jansen",
):
    """
    computes the sensitivity algorithms computed after the Jansen method and exports it to a .pkl
    file

    Parameters:
        ----------
        type_of_metamodel: str
            type of metamodel that has been computed. Possible value: "Kriging"
        training_amount: float
            proportion (between 0 and 1) of the initial data used for training (the remaining data
            are used for testing)
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity
        type_of_Sobol_sensitivity_implementation: str
            type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
            sensitivity algorithm is stored.

    Returns:
        -------
        None

    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Jansen = compute_sensitivity_algo_Jansen(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(type_of_metamodel + '_mechanoadaptation_vs_passive_circular'
        , training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Jansen, complete_pkl_filename_sensitivy_algo)


# MauntzKucherenko#
def compute_sensitivity_algo_MauntzKucherenko(distribution, metamodel, sensitivity_experiment_size):
    """
    computes the sensitivity algorithms computed after the MauntzKucherenko method

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        metamodel: ot class
            metamodel (Kriging) computed in the metamodel_creation.py script and stored in a .pkl
            file
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
        -------
        sensitivityAnalysis: ot class
            sensitivity algorithm computed with the openturns class

    """
    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    # complete_pkl_filename = miu.create_pkl_name(type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount)
    # shuffled_sample, _ = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename)
    # input_sample = shuffled_sample[:, 0:5]
    # output_sample = shuffled_sample[:, -2]
    sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(myExperiment, metamodel, True)
    # sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(input_sample, output_sample, 4096)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_MauntzKucherenko(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="MauntzKucherenko",
):
    """
    computes the sensitivity algorithms computed after the MauntzKucherenko method and exports it
    to a .pkl file

    Parameters:
        ----------
        type_of_metamodel: str
            type of metamodel that has been computed. Possible value: "Kriging"
        training_amount: float
            proportion (between 0 and 1) of the initial data used for training (the remaining data
            are used for testing)
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity
        type_of_Sobol_sensitivity_implementation: str
            type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
            sensitivity algorithm is stored.

    Returns:
        -------
        None

    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_MauntzKucherenko = compute_sensitivity_algo_MauntzKucherenko(
        distribution, metamodel, sensitivity_experiment_size
    )
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(type_of_metamodel + '_mechanoadaptation_vs_passive_circular'
        , training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_MauntzKucherenko, complete_pkl_filename_sensitivy_algo)


# Martinez#
def compute_sensitivity_algo_Martinez(distribution, metamodel, sensitivity_experiment_size):
    """
    computes the sensitivity algorithms computed after the Martinez method

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        metamodel: ot class
            metamodel (Kriging) computed in the metamodel_creation.py script and stored in a .pkl
            file
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
        -------
        sensitivityAnalysis: ot class
            sensitivity algorithm computed with the openturns class

    """

    uniform_distribution = distribution.uniform()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), uniform_distribution, sensitivity_experiment_size, True
    )
    sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(myExperiment, metamodel, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_Martinez(
    type_of_metamodel,
    training_amount,
    distribution,
    sensitivity_experiment_size,
    type_of_Sobol_sensitivity_implementation="Martinez",
):
    """
    computes the sensitivity algorithms computed after the Martinez method and exports it to a .pkl
    file

    Parameters:
        ----------
        type_of_metamodel: str
            type of metamodel that has been computed. Possible value: "Kriging"
        training_amount: float
            proportion (between 0 and 1) of the initial data used for training (the remaining data
            are used for testing)
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity
        type_of_Sobol_sensitivity_implementation: str
            type of Sobol algorithm. Used only to generate the name of the .pkl file in which the
            sensitivity algorithm is stored.

    Returns:
        -------
        None

    """

    complete_pkl_filename_metamodel = miu.create_pkl_name(type_of_metamodel + '_mechanoadaptation_vs_passive_circular', training_amount)
    _, results_from_algo = miu.extract_metamodel_and_data_from_pkl(complete_pkl_filename_metamodel)
    metamodel = metamodelposttreatment.get_metamodel_from_results_algo(results_from_algo)
    sensitivity_algo_Martinez = compute_sensitivity_algo_Martinez(distribution, metamodel, sensitivity_experiment_size)
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(type_of_metamodel + '_mechanoadaptation_vs_passive_circular'
        , training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    miu.export_sensitivity_algo_to_pkl(sensitivity_algo_Martinez, complete_pkl_filename_sensitivy_algo)



#Utils

def get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    ):
    complete_pkl_filename_sensitivy_algo = miu.create_pkl_name_sensitivityalgo(type_of_metamodel + '_mechanoadaptation_vs_passive_circular'
        , training_amount, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation
    )
    sensitivity_algo = miu.extract_sensitivity_algo_from_pkl(complete_pkl_filename_sensitivy_algo)
    first_order_indices = sensitivity_algo.getFirstOrderIndices()
    first_order_indices_confidence_interval = sensitivity_algo.getFirstOrderIndicesInterval()
    first_order_indices_confidence_lowerbounds = [
        first_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    first_order_indices_confidence_upperbounds = [
        first_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    total_order_indices = sensitivity_algo.getTotalOrderIndices()
    total_order_indices_confidence_interval = sensitivity_algo.getTotalOrderIndicesInterval()
    total_order_indices_confidence_lowerbounds = [
        total_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    total_order_indices_confidence_upperbounds = [
        total_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    first_order_indices_confidence_errorbar = np.zeros((2, 5))
    total_order_indices_confidence_errorbar = np.zeros((2, 5))
    for k in range(5):
        first_order_indices_confidence_errorbar[0, k] = (
            first_order_indices[k] - first_order_indices_confidence_lowerbounds[k]
        )
        first_order_indices_confidence_errorbar[1, k] = (
            first_order_indices_confidence_upperbounds[k] - first_order_indices[k]
        )
        total_order_indices_confidence_errorbar[0, k] = (
            total_order_indices[k] - total_order_indices_confidence_lowerbounds[k]
        )
        total_order_indices_confidence_errorbar[1, k] = (
            total_order_indices_confidence_upperbounds[k] - total_order_indices[k]
        )
    return first_order_indices, total_order_indices, first_order_indices_confidence_errorbar, total_order_indices_confidence_errorbar

def gradient(vector):
    nb_samples = np.shape(vector)[0]
    gradient = np.zeros((1, nb_samples-1))
    for k in range(nb_samples-1):
        gradient[0, k] = np.abs(np.diff(vector)[k])/ abs(vector[k])
    return gradient

def compute_convergence_Sobol_indices(
    sensitivity_experiment_size_list,
    type_of_metamodel,
    training_amount,
    distribution,
    createfigure,
    pixels,
):
    Saltelli_first_order_indices_vs_experiment_size = np.zeros((5, len(sensitivity_experiment_size_list)))
    Saltelli_total_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size = np.zeros((2, 5, len(sensitivity_experiment_size_list)))
    Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    Saltelli_computation_time_vs_experiment_size = np.zeros_like(sensitivity_experiment_size_list)
    Jansen_first_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    Jansen_total_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    Jansen_first_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    Jansen_total_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    Jansen_computation_time_vs_experiment_size = np.zeros_like(sensitivity_experiment_size_list)
    MauntzKucherenko_first_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    MauntzKucherenko_total_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    MauntzKucherenko_computation_time_vs_experiment_size = np.zeros_like(sensitivity_experiment_size_list)
    Martinez_first_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    Martinez_total_order_indices_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size)
    Martinez_first_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    Martinez_total_order_indices_confidence_errorbars_vs_experiment_size = np.zeros_like(Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size)
    Martinez_computation_time_vs_experiment_size = np.zeros_like(sensitivity_experiment_size_list)

    for i in range(len(sensitivity_experiment_size_list)):
        sensitivity_experiment_size = int(sensitivity_experiment_size_list[i])
        start_Saltelli = time.time()
        compute_and_export_sensitivity_algo_Saltelli(type_of_metamodel, training_amount, distribution, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation="Saltelli")
        elapsed_Saltelli = time.time() - start_Saltelli
        Saltelli_computation_time_vs_experiment_size[i] = elapsed_Saltelli
        Saltelli_first_order_indices, Saltelli_total_order_indices, Saltelli_first_order_indices_confidence_errorbar, Saltelli_total_order_indices_confidence_errorbar = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, "Saltelli")
        for k in range(5):
            Saltelli_first_order_indices_vs_experiment_size[k, i] = Saltelli_first_order_indices[k]
            Saltelli_total_order_indices_vs_experiment_size[k, i] = Saltelli_total_order_indices[k]
            for j in range(2):
                Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Saltelli_first_order_indices_confidence_errorbar[j, k]
                Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Saltelli_total_order_indices_confidence_errorbar[j, k]
        
        start_Jansen = time.time()
        compute_and_export_sensitivity_algo_Jansen(type_of_metamodel, training_amount, distribution, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation="Jansen")
        elapsed_Jansen = time.time() - start_Jansen
        Jansen_computation_time_vs_experiment_size[i] = elapsed_Jansen
        Jansen_first_order_indices, Jansen_total_order_indices, Jansen_first_order_indices_confidence_errorbar, Jansen_total_order_indices_confidence_errorbar = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, "Jansen")
        for k in range(5):
            Jansen_first_order_indices_vs_experiment_size[k, i] = Jansen_first_order_indices[k]
            Jansen_total_order_indices_vs_experiment_size[k, i] = Jansen_total_order_indices[k]
            for j in range(2):
                Jansen_first_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Jansen_first_order_indices_confidence_errorbar[j, k]
                Jansen_total_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Jansen_total_order_indices_confidence_errorbar[j, k]

        start_MauntzKucherenko = time.time()
        compute_and_export_sensitivity_algo_MauntzKucherenko(type_of_metamodel, training_amount, distribution, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation="MauntzKucherenko")
        elapsed_MauntzKucherenko = time.time() - start_MauntzKucherenko
        MauntzKucherenko_computation_time_vs_experiment_size[i] = elapsed_MauntzKucherenko
        MauntzKucherenko_first_order_indices, MauntzKucherenko_total_order_indices, MauntzKucherenko_first_order_indices_confidence_errorbar, MauntzKucherenko_total_order_indices_confidence_errorbar = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, "MauntzKucherenko")
        for k in range(5):
            MauntzKucherenko_first_order_indices_vs_experiment_size[k, i] = MauntzKucherenko_first_order_indices[k]
            MauntzKucherenko_total_order_indices_vs_experiment_size[k, i] = MauntzKucherenko_total_order_indices[k]
            for j in range(2):
                MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = MauntzKucherenko_first_order_indices_confidence_errorbar[j, k]
                MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = MauntzKucherenko_total_order_indices_confidence_errorbar[j, k]

        start_Martinez = time.time()
        compute_and_export_sensitivity_algo_Martinez(type_of_metamodel, training_amount, distribution, sensitivity_experiment_size, type_of_Sobol_sensitivity_implementation="Martinez")
        elapsed_Martinez = time.time() - start_Martinez
        Martinez_computation_time_vs_experiment_size[i] = elapsed_Martinez
        Martinez_first_order_indices, Martinez_total_order_indices, Martinez_first_order_indices_confidence_errorbar, Martinez_total_order_indices_confidence_errorbar = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, "Martinez")
        for k in range(5):
            Martinez_first_order_indices_vs_experiment_size[k, i] = Martinez_first_order_indices[k]
            Martinez_total_order_indices_vs_experiment_size[k, i] = Martinez_total_order_indices[k]
            for j in range(2):
                Martinez_first_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Martinez_first_order_indices_confidence_errorbar[j, k]
                Martinez_total_order_indices_confidence_errorbars_vs_experiment_size[j, k, i] = Martinez_total_order_indices_confidence_errorbar[j, k]

    miu.export_utils_sensitivity_convergence("sensitivity_analysis_convergence_mechanoadaptation_vs_passive_circular.pkl", Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size)

    Saltelli_first_order_indices_vs_experiment_size_confinterval = np.zeros((5, len(sensitivity_experiment_size_list)))
    Saltelli_total_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    Jansen_first_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    Jansen_total_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    Martinez_first_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    Martinez_total_order_indices_vs_experiment_size_confinterval = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_confinterval)
    for k in range(5):
        Saltelli_first_order_indices_vs_experiment_size_confinterval[k, :] = Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        Saltelli_total_order_indices_vs_experiment_size_confinterval[k, :] = Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        Jansen_first_order_indices_vs_experiment_size_confinterval[k, :] = Jansen_first_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Jansen_first_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        Jansen_total_order_indices_vs_experiment_size_confinterval[k, :] = Jansen_total_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Jansen_total_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval[k, :] = MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval[k, :] = MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        Martinez_first_order_indices_vs_experiment_size_confinterval[k, :] = Martinez_first_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Martinez_first_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]
        Martinez_total_order_indices_vs_experiment_size_confinterval[k, :] = Martinez_total_order_indices_confidence_errorbars_vs_experiment_size[0, k, :] + Martinez_total_order_indices_confidence_errorbars_vs_experiment_size[1, k, :]


    miu.export_utils_sensitivity_convergence_confinterval("sensitivity_analysis_convergence_confinterval_mechanoadaptation_vs_passive_circular.pkl", Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval)

    Saltelli_first_order_indices_vs_experiment_size_gradient = np.zeros((5, len(sensitivity_experiment_size_list)-1))
    Saltelli_total_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    Jansen_first_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    Jansen_total_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    MauntzKucherenko_first_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    MauntzKucherenko_total_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    Martinez_first_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    Martinez_total_order_indices_vs_experiment_size_gradient = np.zeros_like(Saltelli_first_order_indices_vs_experiment_size_gradient)
    for k in range(5):
        Saltelli_first_order_indices_vs_experiment_size_gradient[k, :] = gradient(Saltelli_first_order_indices_vs_experiment_size[k, :])
        Saltelli_total_order_indices_vs_experiment_size_gradient[k, :] = gradient(Saltelli_total_order_indices_vs_experiment_size[k, :])
        Jansen_first_order_indices_vs_experiment_size_gradient[k, :] = gradient(Jansen_first_order_indices_vs_experiment_size[k, :])
        Jansen_total_order_indices_vs_experiment_size_gradient[k, :] = gradient(Jansen_total_order_indices_vs_experiment_size[k, :])
        MauntzKucherenko_first_order_indices_vs_experiment_size_gradient[k, :] = gradient(MauntzKucherenko_first_order_indices_vs_experiment_size[k, :])
        MauntzKucherenko_total_order_indices_vs_experiment_size_gradient[k, :] = gradient(MauntzKucherenko_total_order_indices_vs_experiment_size[k, :])
        Martinez_first_order_indices_vs_experiment_size_gradient[k, :] = gradient(Martinez_first_order_indices_vs_experiment_size[k, :])
        Martinez_total_order_indices_vs_experiment_size_gradient[k, :] = gradient(Martinez_total_order_indices_vs_experiment_size[k, :])
        

    miu.export_utils_sensitivity_convergence_gradient("sensitivity_analysis_convergence_gradient_mechanoadaptation_vs_passive_circular.pkl", Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient)



# Sobol indices from PCE#
def compute_sensitivity_indices_PCE(training_amount, degree):
    """Computes the sensitivity algorithms directly from the coefficients of the PCE metamodel

    Parameters:
    ----------
    training_amount: float
        Proportion (between 0 and 1) of the initial data used for training (the remaining data are
        used for testing)
    degree: float
        truncature degree of the PCE metamodel

    Returns:
    -------
    first_order_indices: list
        First order Sobol indices. Position in the list follows the position in the datafile:
        gamma_bar_r, gamma_bar_fs, gamma_bar_lambda
    total_order_indices: list
        Total order Sobol indices. Position in the list follows the position in the datafile:
        gamma_bar_r, gamma_bar_fs, gamma_bar_lambda
    """

    complete_filename = miu.create_pkl_name("PCE_mechanoadaptation_vs_passive_circular" + str(degree), training_amount, folder="")
    [_, results_from_algo] = miu.extract_metamodel_and_data_from_pkl(complete_filename)
    chaosSI = ot.FunctionalChaosSobolIndices(results_from_algo)
    first_order_indices = [chaosSI.getSobolIndex(k) for k in range(5)]
    total_order_indices = [chaosSI.getSobolTotalIndex(k) for k in range(5)]
    return first_order_indices, total_order_indices


# PLOTS#

def plot_convergence_Sobol_indices(sensitivity_experiment_size_list, type_of_Sobol_sensitivity_implementation, first_order_indices_vs_experiment_size,
total_order_indices_vs_experiment_size,
first_order_indices_confidence_errorbars_vs_experiment_size,
total_order_indices_confidence_errorbars_vs_experiment_size,
computation_time_vs_experiment_size,     createfigure,
    pixels, savefigure):

    fig_first = createfigure.rectangle_figure(pixels=pixels)
    ax_first = fig_first.gca()
    fig_total = createfigure.rectangle_figure(pixels=pixels)
    ax_total = fig_total.gca()
    labels = [r"$\overline{\gamma}_0$", r"$\overline{\sigma}$", r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_S$"]
    palette = sns.color_palette("Paired")
    # palette = sns.color_palette("husl", 6)

    # errorbar_colors = [palette[k] for k in range(0, len(palette), 2)]
    line_colors = [palette[k] for k in range(1, len(palette), 2)]
    ax_first.set_xscale('log')
    for i in range(5):
        ax_first.plot(
            sensitivity_experiment_size_list,
            first_order_indices_vs_experiment_size[i, :],
            # yerr=first_order_indices_confidence_errorbars_vs_experiment_size[:, i, :],
            label=labels[i],
            color=line_colors[i], 
            # ecolor=errorbar_colors[i],
            # ealpha=0.7,
            # elinewidth=0.5,
            lw=2)
        ax_first.fill_between(sensitivity_experiment_size_list, first_order_indices_vs_experiment_size[i, :] + first_order_indices_confidence_errorbars_vs_experiment_size[0, i, :], first_order_indices_vs_experiment_size[i, :]- first_order_indices_confidence_errorbars_vs_experiment_size[1, i, :], color=line_colors[i], alpha = 0.2)


    # secax_first=ax_first.twinx()
    # secax_first.plot(sensitivity_experiment_size_list, computation_time_vs_experiment_size, ':k')
    # secax_total=ax_total.twinx()
    # secax_total.plot(sensitivity_experiment_size_list, computation_time_vs_experiment_size, ':k')
    ax_total.set_xscale('log')
    for i in range(5):
        ax_total.plot(
            sensitivity_experiment_size_list,
            total_order_indices_vs_experiment_size[i, :],
            # yerr=total_order_indices_confidence_errorbars_vs_experiment_size[:, i, :],
            label=labels[i],
            color=line_colors[i], 
            # ecolor=errorbar_colors[i],
            # ealpha=0.7,
            # elinewidth=0.5,
            lw=2)
        ax_total.fill_between(sensitivity_experiment_size_list, total_order_indices_vs_experiment_size[i, :] + total_order_indices_confidence_errorbars_vs_experiment_size[0, i, :], total_order_indices_vs_experiment_size[i, :]- total_order_indices_confidence_errorbars_vs_experiment_size[1, i, :], color=line_colors[i], alpha = 0.2)


    ax_first.set_ylim((-0.7, 1.3))
    ax_first.set_xlim((10, 1.5*1e5))
    ax_first.set_xticks([10, 100, 1000, 10000, 100000])
    ax_first.set_xticklabels(
        ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax_first.set_yticks([-0.6, -0.4,  -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_first.set_yticklabels(
        [-0.6, -0.4,  -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )

    # secax_first.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # secax_first.set_yticklabels(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    # secax_total.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # secax_total.set_yticklabels(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    ax_total.set_xlim((10, 1.5*1e5))
    ax_total.set_ylim((-0.5, 1.5))
    # secax_first.set_ylim((-0.2, 8.2))
    # secax_total.set_ylim((-0.2, 8.2))
    ax_total.set_xticks([10, 100, 1000, 10000, 100000])
    ax_total.set_xticklabels(
        ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax_total.set_yticks([-0.4,  -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4])
    ax_total.set_yticklabels(
        [-0.4,  -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )

    ax_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)

    ax_first.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax_total.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax_first.set_ylabel("$S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax_total.set_ylabel("$ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    # secax_first.set_ylabel("Computation time [ s ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    # secax_total.set_ylabel("Computation time [ s ]", font=fonts.serif(), fontsize=fonts.axis_label_size())

    savefigure.save_as_png(fig_first, 'convergence_firstSobol_mechanoadaptation_vs_passive_circular_' +  type_of_Sobol_sensitivity_implementation + str(pixels) + "p")
    savefigure.save_as_png(fig_total, 'convergence_totalSobol_mechanoadaptation_vs_passive_circular_' +  type_of_Sobol_sensitivity_implementation + str(pixels) + "p")


def plot_confinterval_Sobol_vs_sample_size(sensitivity_experiment_size_list):
    """
    Plots the absolute confinterval of the cumulative mean of a sample

    Parameters:
        ----------
        None

    Returns:
        -------
        None

    """
    filename_sensitivity_convergence_confinterval = "sensitivity_analysis_convergence_confinterval_mechanoadaptation_vs_passive_circular.pkl"
    Saltelli_first_order_indices_vs_experiment_size_confinterval,Saltelli_total_order_indices_vs_experiment_size_confinterval,Jansen_first_order_indices_vs_experiment_size_confinterval,Jansen_total_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval,MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval,Martinez_first_order_indices_vs_experiment_size_confinterval,Martinez_total_order_indices_vs_experiment_size_confinterval = miu.extract_sensitivity_convergence_confinterval(filename_sensitivity_convergence_confinterval)

    labels = [r"$\overline{\gamma}_0$", r"$\overline{\sigma}$", r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_S$"]
    palette = sns.color_palette("Paired")
    line_colors = [palette[k] for k in range(1, len(palette), 2)]

    def setting_ax(ax):
        ax.grid(linestyle=':')
        ax.set_xscale('log')
        ax.plot(sensitivity_experiment_size_list, [0.05] * len(sensitivity_experiment_size_list), "--k")
        ax.set_xlim((10, 1e5))
        ax.set_xticks([10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(
            ["$10^1$", "$10^2$", "$10^3$", "$10^4$","$10^5$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(9e-4, 5)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
        ax.set_yticklabels(
            ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())


    fig_Saltelli_first = createfigure.rectangle_figure(pixels)
    ax_Saltelli_first = fig_Saltelli_first.gca()
    setting_ax(ax_Saltelli_first)
    for i in range(5):
        ax_Saltelli_first.plot(
            sensitivity_experiment_size_list,
            Saltelli_first_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Saltelli_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Saltelli_first.set_ylabel(r"Range of the CIs of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Saltelli_first, "confinterval_firstSobol_Saltelli_mechanoadaptation_vs_passive_circular")

    fig_Saltelli_total = createfigure.rectangle_figure(pixels)
    ax_Saltelli_total = fig_Saltelli_total.gca()
    setting_ax(ax_Saltelli_total)
    for i in range(5):
        ax_Saltelli_total.plot(
            sensitivity_experiment_size_list,
            Saltelli_total_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Saltelli_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Saltelli_total.set_ylabel(r"Range of the CIs of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Saltelli_total, "confinterval_totalSobol_Saltelli_mechanoadaptation_vs_passive_circular")

    fig_Jansen_first = createfigure.rectangle_figure(pixels)
    ax_Jansen_first = fig_Jansen_first.gca()
    setting_ax(ax_Jansen_first)
    for i in range(5):
        ax_Jansen_first.plot(
            sensitivity_experiment_size_list,
            Jansen_first_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Jansen_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Jansen_first.set_ylabel(r"Range of the CIs of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Jansen_first, "confinterval_firstSobol_Jansen_mechanoadaptation_vs_passive_circular")

    fig_Jansen_total = createfigure.rectangle_figure(pixels)
    ax_Jansen_total = fig_Jansen_total.gca()
    setting_ax(ax_Jansen_total)
    for i in range(5):
        ax_Jansen_total.plot(
            sensitivity_experiment_size_list,
            Jansen_total_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Jansen_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Jansen_total.set_ylabel(r"Range of the CIs of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Jansen_total, "confinterval_totalSobol_Jansen_mechanoadaptation_vs_passive_circular")

    fig_MauntzKucherenko_first = createfigure.rectangle_figure(pixels)
    ax_MauntzKucherenko_first = fig_MauntzKucherenko_first.gca()
    setting_ax(ax_MauntzKucherenko_first)
    for i in range(5):
        ax_MauntzKucherenko_first.plot(
            sensitivity_experiment_size_list,
            MauntzKucherenko_first_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_MauntzKucherenko_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_MauntzKucherenko_first.set_ylabel(r"Range of the CIs of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_MauntzKucherenko_first, "confinterval_firstSobol_MauntzKucherenko_mechanoadaptation_vs_passive_circular")

    fig_MauntzKucherenko_total = createfigure.rectangle_figure(pixels)
    ax_MauntzKucherenko_total = fig_MauntzKucherenko_total.gca()
    setting_ax(ax_MauntzKucherenko_total)
    for i in range(5):
        ax_MauntzKucherenko_total.plot(
            sensitivity_experiment_size_list,
            MauntzKucherenko_total_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_MauntzKucherenko_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_MauntzKucherenko_total.set_ylabel(r"Range of the CIs of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_MauntzKucherenko_total, "confinterval_totalSobol_MauntzKucherenko_mechanoadaptation_vs_passive_circular")

    fig_Martinez_first = createfigure.rectangle_figure(pixels)
    ax_Martinez_first = fig_Martinez_first.gca()
    setting_ax(ax_Martinez_first)
    for i in range(5):
        ax_Martinez_first.plot(
            sensitivity_experiment_size_list,
            Martinez_first_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Martinez_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Martinez_first.set_ylabel(r"Range of the CIs of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Martinez_first, "confinterval_firstSobol_Martinez_mechanoadaptation_vs_passive_circular")

    fig_Martinez_total = createfigure.rectangle_figure(pixels)
    ax_Martinez_total = fig_Martinez_total.gca()
    setting_ax(ax_Martinez_total)
    for i in range(5):
        ax_Martinez_total.plot(
            sensitivity_experiment_size_list,
            Martinez_total_order_indices_vs_experiment_size_confinterval[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Martinez_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Martinez_total.set_ylabel(r"Range of the CIs of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Martinez_total, "confinterval_totalSobol_Martinez_mechanoadaptation_vs_passive_circular")


def plot_gradient_Sobol_vs_sample_size(sensitivity_experiment_size_list):
    """
    Plots the absolute gradient of the cumulative mean of a sample

    Parameters:
        ----------
        None

    Returns:
        -------
        None

    """
    filename_sensitivity_convergence_gradient = "sensitivity_analysis_convergence_gradient_mechanoadaptation_vs_passive_circular.pkl"
    Saltelli_first_order_indices_vs_experiment_size_gradient,Saltelli_total_order_indices_vs_experiment_size_gradient,Jansen_first_order_indices_vs_experiment_size_gradient,Jansen_total_order_indices_vs_experiment_size_gradient,MauntzKucherenko_first_order_indices_vs_experiment_size_gradient,MauntzKucherenko_total_order_indices_vs_experiment_size_gradient,Martinez_first_order_indices_vs_experiment_size_gradient,Martinez_total_order_indices_vs_experiment_size_gradient = miu.extract_sensitivity_convergence_gradient(filename_sensitivity_convergence_gradient)

    labels = [r"$\overline{\gamma}_0$", r"$\overline{\sigma}$", r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_S$"]
    palette = sns.color_palette("Paired")
    line_colors = [palette[k] for k in range(1, len(palette), 2)]

    def setting_ax(ax):
        ax.grid(linestyle=':')
        ax.set_xscale('log')
        ax.plot(sensitivity_experiment_size_list, [0.01] * len(sensitivity_experiment_size_list), "--k")
        ax.set_xlim((10, 1e5))
        ax.set_xticks([10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(
            ["$10^1$", "$10^2$", "$10^3$", "$10^4$","$10^5$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 5*1e-1)
        ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(
            ["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"],
            font=fonts.serif(),
            fontsize=fonts.axis_legend_size(),
        )   
        ax.set_xlabel("Number of samples [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())


    fig_Saltelli_first = createfigure.rectangle_figure(pixels)
    ax_Saltelli_first = fig_Saltelli_first.gca()
    setting_ax(ax_Saltelli_first)
    for i in range(5):
        ax_Saltelli_first.plot(
            sensitivity_experiment_size_list[1:],
            Saltelli_first_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Saltelli_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Saltelli_first.set_ylabel(r"Grad of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Saltelli_first, "gradient_firstSobol_Saltelli_mechanoadaptation_vs_passive_circular")

    fig_Saltelli_total = createfigure.rectangle_figure(pixels)
    ax_Saltelli_total = fig_Saltelli_total.gca()
    setting_ax(ax_Saltelli_total)
    for i in range(5):
        ax_Saltelli_total.plot(
            sensitivity_experiment_size_list[1:],
            Saltelli_total_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Saltelli_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Saltelli_total.set_ylabel(r"Grad of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Saltelli_total, "gradient_totalSobol_Saltelli_mechanoadaptation_vs_passive_circular")

    fig_Jansen_first = createfigure.rectangle_figure(pixels)
    ax_Jansen_first = fig_Jansen_first.gca()
    setting_ax(ax_Jansen_first)
    for i in range(5):
        ax_Jansen_first.plot(
            sensitivity_experiment_size_list[1:],
            Jansen_first_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Jansen_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Jansen_first.set_ylabel(r"Grad of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Jansen_first, "gradient_firstSobol_Jansen_mechanoadaptation_vs_passive_circular")

    fig_Jansen_total = createfigure.rectangle_figure(pixels)
    ax_Jansen_total = fig_Jansen_total.gca()
    setting_ax(ax_Jansen_total)
    for i in range(5):
        ax_Jansen_total.plot(
            sensitivity_experiment_size_list[1:],
            Jansen_total_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Jansen_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Jansen_total.set_ylabel(r"Grad of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Jansen_total, "gradient_totalSobol_Jansen_mechanoadaptation_vs_passive_circular")

    fig_MauntzKucherenko_first = createfigure.rectangle_figure(pixels)
    ax_MauntzKucherenko_first = fig_MauntzKucherenko_first.gca()
    setting_ax(ax_MauntzKucherenko_first)
    for i in range(5):
        ax_MauntzKucherenko_first.plot(
            sensitivity_experiment_size_list[1:],
            MauntzKucherenko_first_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_MauntzKucherenko_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_MauntzKucherenko_first.set_ylabel(r"Grad of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_MauntzKucherenko_first, "gradient_firstSobol_MauntzKucherenko_mechanoadaptation_vs_passive_circular")

    fig_MauntzKucherenko_total = createfigure.rectangle_figure(pixels)
    ax_MauntzKucherenko_total = fig_MauntzKucherenko_total.gca()
    setting_ax(ax_MauntzKucherenko_total)
    for i in range(5):
        ax_MauntzKucherenko_total.plot(
            sensitivity_experiment_size_list[1:],
            MauntzKucherenko_total_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_MauntzKucherenko_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_MauntzKucherenko_total.set_ylabel(r"Grad of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_MauntzKucherenko_total, "gradient_totalSobol_MauntzKucherenko_mechanoadaptation_vs_passive_circular")

    fig_Martinez_first = createfigure.rectangle_figure(pixels)
    ax_Martinez_first = fig_Martinez_first.gca()
    setting_ax(ax_Martinez_first)
    for i in range(5):
        ax_Martinez_first.plot(
            sensitivity_experiment_size_list[1:],
            Martinez_first_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Martinez_first.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Martinez_first.set_ylabel(r"Grad of $S_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Martinez_first, "gradient_firstSobol_Martinez_mechanoadaptation_vs_passive_circular")

    fig_Martinez_total = createfigure.rectangle_figure(pixels)
    ax_Martinez_total = fig_Martinez_total.gca()
    setting_ax(ax_Martinez_total)
    for i in range(5):
        ax_Martinez_total.plot(
            sensitivity_experiment_size_list[1:],
            Martinez_total_order_indices_vs_experiment_size_gradient[i, :],
            label=labels[i],
            color=line_colors[i], 
            lw=2)
    ax_Martinez_total.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    ax_Martinez_total.set_ylabel(r"Grad of $ST_i$ [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig_Martinez_total, "gradient_totalSobol_Martinez_mechanoadaptation_vs_passive_circular")


def plot_comparison_indices(
    training_amount,
    degree,
    sensitivity_experiment_size,
    createfigure,
    pixels,
    
):
    first_order_indices_PCE, total_order_indices_PCE = compute_sensitivity_indices_PCE(training_amount, degree)
    first_order_indices_MauntzKucherenko, total_order_indices_MauntzKucherenko, first_order_indices_confidence_errorbar_MauntzKucherenko, total_order_indices_confidence_errorbar_MauntzKucherenko = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, 'MauntzKucherenko')
    first_order_indices_Saltelli, total_order_indices_Saltelli, first_order_indices_confidence_errorbar_Saltelli, total_order_indices_confidence_errorbar_Saltelli = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, 'Saltelli')
    first_order_indices_Jansen, total_order_indices_Jansen, first_order_indices_confidence_errorbar_Jansen, total_order_indices_confidence_errorbar_Jansen = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, 'Jansen')
    first_order_indices_Martinez, total_order_indices_Martinez, first_order_indices_confidence_errorbar_Martinez, total_order_indices_confidence_errorbar_Martinez = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, 'Martinez')
    palette_initial = sns.color_palette("Paired")
    palette = [palette_initial[k] for k in range(1, len(palette_initial), 2)]
    color_MauntzKucherenko = palette[0]
    color_Saltelli = palette[1]
    color_Jansen = palette[2]
    color_Martinez = palette[3]
    color_PCE = 'k'
    alpha_Kriging = 0.6
    alpha_PCE = 0.99
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.errorbar(
        [0, 1, 2, 3, 4],
        first_order_indices_MauntzKucherenko,
        yerr=first_order_indices_confidence_errorbar_MauntzKucherenko,
        label="Mauntz-Kucherenko",
        color=color_MauntzKucherenko,
        alpha = alpha_Kriging,
        marker="v",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2, 3, 4],
        total_order_indices_MauntzKucherenko,
        yerr=total_order_indices_confidence_errorbar_MauntzKucherenko,
        color=color_MauntzKucherenko,
        alpha = alpha_Kriging,
        marker="D",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )

    ax.errorbar(
        [0, 1, 2, 3, 4],
        first_order_indices_Saltelli,
        yerr=first_order_indices_confidence_errorbar_Saltelli,
        label="Saltelli",
        color=color_Saltelli,
        alpha = alpha_Kriging,
        marker="v",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2, 3, 4],
        total_order_indices_Saltelli,
        yerr=total_order_indices_confidence_errorbar_Saltelli,
        color=color_Saltelli,
        alpha = alpha_Kriging,
        marker="D",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )

    ax.errorbar(
        [0, 1, 2, 3, 4],
        first_order_indices_Jansen,
        yerr=first_order_indices_confidence_errorbar_Jansen,
        label="Jansen",
        color=color_Jansen,
        alpha = alpha_Kriging,
        marker="v",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2, 3, 4],
        total_order_indices_Jansen,
        yerr=total_order_indices_confidence_errorbar_Jansen,
        color=color_Jansen,
        alpha = alpha_Kriging,
        marker="D",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )

    ax.errorbar(
        [0, 1, 2, 3, 4],
        first_order_indices_Martinez,
        yerr=first_order_indices_confidence_errorbar_Martinez,
        label="Martinez",
        color=color_Martinez,
        alpha = alpha_Kriging,
        marker="v",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2, 3, 4],
        total_order_indices_Martinez,
        yerr=total_order_indices_confidence_errorbar_Martinez,
        color=color_Martinez,
        alpha = alpha_Kriging,
        marker="D",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )

    ax.plot(
        [0, 1, 2, 3, 4],
        first_order_indices_PCE,
        label="PCE",
        color=color_PCE,
        alpha = alpha_PCE,
        marker="v",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )
    ax.plot(
        [0, 1, 2, 3, 4],
        total_order_indices_PCE,
        color=color_PCE,
        alpha = alpha_PCE,
        marker="D",
        markerfacecolor='none',
        markersize=12,
        linestyle="None",
    )



    ax.set_xlim((-0.2, 4.2))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(
        [r"$\overline{\gamma}_0$", r"$\overline{\sigma}$", r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_S$"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(
        ["0", "0.25", "0.5", "0.75", "1"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlabel("Variable", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Sobol indices [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    filename = (
        "sobolindices_algo_COMPARISON"
        + "_mechanoadaptation_vs_passive_circular_trainingamount="
        + str(training_amount)
        + "_size="
        + str(sensitivity_experiment_size)
        + "_"
    )
    savefigure.save_as_png(fig, filename + str(pixels) + "p")


def plot_comparison_indices_piechart(
    training_amount,
    sensitivity_experiment_size,
    createfigure,
    pixels,
    
):
    _, total_order_indices_MauntzKucherenko, _, _ = get_indices_and_confidence_intervals(type_of_metamodel, training_amount, sensitivity_experiment_size, 'MauntzKucherenko')
    palette = sns.color_palette("Paired")
    colors = (palette[k] for k in range(1, len(palette), 2))
    labels = [r"$\overline{\gamma}_0$", r"$\overline{\sigma}$", r"$\overline{\gamma}_A$", r"$\overline{\gamma}_D$", r"$\overline{\gamma}_S$"]
    fig = createfigure.square_figure_7(pixels=pixels)
    ax = fig.gca()
    ax.pie(total_order_indices_MauntzKucherenko, labels = labels, labeldistance=0.5, colors = colors, wedgeprops={'alpha':0.6}, textprops = dict(family="serif", weight="normal", style="normal", size=24))
    filename = ('sobolindices_Kriging_piechart_mechanoadaptation_vs_passive_circular')
    savefigure.save_as_png(fig, filename + str(pixels) + "p")


if __name__ == "__main__":
    type_of_metamodel = "Kriging"
    training_amount = 0.8
    sensitivity_experiment_size_list = np.array([10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 100000])#np.arange(10, 20000, 100)

    type_of_Sobol_sensitivity_implementation_list = ["MauntzKucherenko", "Saltelli", "Jansen", "MauntzKucherenko", "Martinez"]

    metamodelposttreatment = MetamodelPostTreatment()
    distribution = Distribution()
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    pixels = 360

    degree = 7

    compute_convergence_Sobol_indices(
        sensitivity_experiment_size_list,
        type_of_metamodel,
        training_amount,
        distribution,
        createfigure,
        pixels)

    filename_sensitivity_convergence = "sensitivity_analysis_convergence_mechanoadaptation_vs_passive_circular.pkl"
    Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size = miu.extract_sensitivity_convergence(filename_sensitivity_convergence)


    plot_convergence_Sobol_indices(sensitivity_experiment_size_list, 'Martinez', Martinez_first_order_indices_vs_experiment_size, Martinez_total_order_indices_vs_experiment_size, Martinez_first_order_indices_confidence_errorbars_vs_experiment_size, Martinez_total_order_indices_confidence_errorbars_vs_experiment_size, Martinez_computation_time_vs_experiment_size, createfigure, pixels, savefigure)
    plot_convergence_Sobol_indices(sensitivity_experiment_size_list, 'Saltelli', Saltelli_first_order_indices_vs_experiment_size, Saltelli_total_order_indices_vs_experiment_size, Saltelli_first_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_total_order_indices_confidence_errorbars_vs_experiment_size, Saltelli_computation_time_vs_experiment_size, createfigure, pixels, savefigure)
    plot_convergence_Sobol_indices(sensitivity_experiment_size_list, 'Jansen', Jansen_first_order_indices_vs_experiment_size, Jansen_total_order_indices_vs_experiment_size, Jansen_first_order_indices_confidence_errorbars_vs_experiment_size, Jansen_total_order_indices_confidence_errorbars_vs_experiment_size, Jansen_computation_time_vs_experiment_size, createfigure, pixels, savefigure)
    plot_convergence_Sobol_indices(sensitivity_experiment_size_list, 'MauntzKucherenko', MauntzKucherenko_first_order_indices_vs_experiment_size, MauntzKucherenko_total_order_indices_vs_experiment_size, MauntzKucherenko_first_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_total_order_indices_confidence_errorbars_vs_experiment_size, MauntzKucherenko_computation_time_vs_experiment_size, createfigure, pixels, savefigure)
    
    plot_confinterval_Sobol_vs_sample_size(sensitivity_experiment_size_list)
    plot_gradient_Sobol_vs_sample_size(sensitivity_experiment_size_list)

    plot_comparison_indices(
    training_amount,
    degree,
    100000,
    createfigure,
    pixels)

    plot_comparison_indices_piechart(
        training_amount,
        100000,
        createfigure,
        pixels,
        
    )