
# ----------------------------------- SMOTEBoost--------------------------------------
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from imblearn.over_sampling import SMOTE
# import numpy as np
# import pandas as pd
#
# # SMOTEBoost implementation
# class SMOTEBoost:
#     def __init__(self, base_estimator=None, n_estimators=50, random_state=None):
#         self.n_estimators = n_estimators
#         self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
#         self.random_state = random_state
#         self.models = []
#         self.alphas = []
#
#     def fit(self, X, y):
#         n_samples, _ = X.shape
#         sample_weights = np.ones(n_samples) / n_samples
#         for i in range(self.n_estimators):
#             # Apply SMOTE to create a balanced dataset
#             smote = SMOTE(sampling_strategy='auto', random_state=self.random_state)
#             X_res, y_res = smote.fit_resample(X, y)
#
#             # Train the weak learner on the resampled dataset
#             model = AdaBoostClassifier(base_estimator=self.base_estimator, n_estimators=1, algorithm='SAMME')
#             model.fit(X_res, y_res)
#             # sample_weights= model.feature_importances_
#
#             # Compute error and alpha
#             y_pred = model.predict(X)
#             err = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)
#             alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
#
#             # Update sample weights
#             sample_weights *= np.exp(-alpha * y * (2 * (y_pred == y) - 1))
#             sample_weights /= np.sum(sample_weights)
#
#             self.models.append(model)
#             self.alphas.append(alpha)
#
#     def predict(self, X):
#         # Aggregate predictions from all weak learners
#         pred = sum(alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models))
#         return np.sign(pred)
#
#
# n_samples = 1000
# n_features = 10
# # Simulated data
# X = np.random.rand(n_samples, n_features)
# y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # Imbalanced data
#
# # Split into training and test sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
#
#
# # Instantiate and train SMOTEBoost
# smoteboost = SMOTEBoost(n_estimators=50, random_state=42)
# smoteboost.fit(X, y)
#
# # Evaluate the model
# y_pred = smoteboost.predict(X)
# print("Accuracy:", accuracy_score(y, y_pred))
# print(classification_report(y, y_pred))

# ----------------------------------------------------------------------------------------


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from mealpy import FloatVar
from mealpy.swarm_based.FFA import OriginalFFA as FFA


class FFA_SMOTE:
    """
    Class for optimizing SMOTE parameters using Firefly Algorithm (FFA).

    Attributes:
        epoch (int): Maximum number of FFA iterations.
        pop_size (int): Population size for FFA.
    """

    def __init__(self, epoch: int = 1, pop_size: int = 5):
        """
        Initialize the FFA_SMOTE class.

        Args:
            epoch (int): Number of iterations for FFA.
            pop_size (int): Number of agents in FFA.
        """
        self.epoch = epoch
        self.pop_size = pop_size
        self.best_params = None  # Store the best parameters
        self.X_res = None  # Store the resampled features
        self.y_res = None  # Store the resampled labels

    def optimize_smote(self, X_train, y_train, X_val, y_val):
        """
        Optimize SMOTE parameters using the Firefly Algorithm.

        Args:
            X_train (array): Training feature set.
            y_train (array): Training labels.
            X_val (array): Validation feature set.
            y_val (array): Validation labels.

        Returns:
            tuple: Best SMOTE parameters and the corresponding F1 score.
        """

        def objective_function(solution):
            """
            Objective function to optimize using FFA.

            Args:
                solution (list): A solution containing k_neighbors and sampling_strategy.

            Returns:
                float: Negative F1 score (to minimize).
            """
            k_neighbors = int(solution[0])  # Convert to integer for SMOTE
            sampling_strategy = solution[1]  # Sampling strategy as a float

            # Apply SMOTE with the current parameters
            smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            # Train a classifier (AdaBoost in this case)
            model = AdaBoostClassifier(n_estimators=50, random_state=42)
            model.fit(X_res, y_res)

            # Validate the model
            y_pred = model.predict(X_val)
            return -f1_score(y_val, y_pred)  # Minimize negative F1 score

        # Define problem bounds and variables
        problem_dict = {
            "obj_func": objective_function,
            # "lb": [2, 0.5],  # Lower bounds: k_neighbors (int), sampling_strategy (float)
            # "ub": [10, 1.0],  # Upper bounds: k_neighbors, sampling_strategy
            "minmax": "min",  # Minimize the objective function
            "verbose": False,
            "bounds": FloatVar(lb=([2, 0.5]) * 30, ub=([10, 1.0]) * 30, name="delta"),
        }

        # Initialize and solve using FFA
        model = FFA(epoch=self.epoch, pop_size=self.pop_size, gamma=0.001, beta_base=2, alpha=0.2,
                    alpha_damp=0.99, delta=0.05, exponent=2)
        best_solution = model.solve(problem_dict)

        # Extract best parameters and corresponding F1 score
        self.best_params = best_solution.solution
        best_f1 = -best_solution.target.fitness

        # Apply SMOTE with optimized parameters and store balanced data
        smote = SMOTE(k_neighbors=int(self.best_params[0]), sampling_strategy=self.best_params[1], random_state=42)
        self.X_res, self.y_res = smote.fit_resample(X_train, y_train)

        return self.best_params, best_f1

    def get_balanced_data(self):
        """
        Return the balanced dataset after SMOTE optimization.

        Returns:
            tuple: Resampled features and labels (X_res, y_res).
        """
        return self.X_res, self.y_res



if __name__ == "__main__":
    # Generate synthetic imbalanced dataset
    np.random.seed(42)
    X = np.random.rand(1000, 20)
    y = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Optimize SMOTE parameters using FFA
    ffa_smote = FFA_SMOTE(epoch=50, pop_size=20)
    best_params, best_f1 = ffa_smote.optimize_smote(X_train, y_train, X_val, y_val)
    print("Optimized SMOTE Parameters:", best_params)
    print("Best Validation F1-Score:", best_f1)

    # Get balanced data
    X_res, y_res = ffa_smote.get_balanced_data()
    print("Balanced Classes Distribution:", dict(zip(*np.unique(y_res, return_counts=True))))

    # Train and test the final model
    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print("Test F1-Score:", test_f1)
