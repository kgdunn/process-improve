=====================================
Multivariate Analysis Methods
=====================================

Overview
========

The ``process_improve.multivariate`` module provides sophisticated multivariate analysis methods specifically designed for process improvement and data analysis in industrial settings. This module includes implementations of Principal Component Analysis (PCA), Partial Least Squares (PLS), and T-shaped Partial Least Squares (TPLS) regression with advanced features like missing data handling and comprehensive diagnostic tools.

Multivariate analysis is essential in process industries where:

* Multiple correlated variables need simultaneous analysis
* Data reduction and visualization are required 
* Predictive models must handle high-dimensional data
* Process monitoring and fault detection are critical
* Missing data is common in real industrial datasets

Available Methods
=================

The module provides three main multivariate analysis methods:

**Principal Component Analysis (PCA)**
    Unsupervised dimensionality reduction technique that identifies the principal directions of variance in the data. Ideal for data exploration, visualization, and process monitoring.

**Partial Least Squares (PLS)**
    Supervised regression method that finds latent variables that maximize covariance between predictor variables (X) and response variables (Y). Excellent for prediction when X variables are highly correlated.

**T-shaped Partial Least Squares (TPLS)**
    Advanced method for analyzing T-shaped data structures common in batch processes, where data is organized in multiple blocks (D, F, Z, Y) representing different aspects of the process.

When to Use Each Method
======================

**Use PCA when:**

* You need to reduce dimensionality for visualization
* You want to identify the main sources of variation
* You need process monitoring with control charts
* You have primarily unsupervised learning tasks
* You want to detect outliers and anomalies

**Use PLS when:**

* You have predictor variables (X) and response variables (Y)
* X variables are highly correlated (multicollinearity)
* You need predictive models with good interpretability
* You want to handle more X variables than observations
* You need robust regression in high-dimensional spaces

**Use TPLS when:**

* You have T-shaped or batch process data
* Data is naturally organized in multiple blocks
* You need to model complex relationships between different data types
* You have time-varying batch processes
* Traditional PLS doesn't capture the data structure adequately

Common Workflows and Best Practices
===================================

**Data Preprocessing**
    Always center and scale your data appropriately. The module provides ``center()`` and ``scale()`` functions, or use the ``MCUVScaler`` class for mean centering and unit variance scaling.

**Missing Data**
    The PCA and PLS implementations support various missing data algorithms including TSR (Trimmed Scores Regression), NIPALS, and SCP methods.

**Model Selection**
    Use cross-validation to determine the optimal number of components. Start with a small number and increase until the predictive performance plateaus.

**Diagnostics**
    Always examine diagnostic plots including score plots, loading plots, Hotelling's T² charts, and SPE (Squared Prediction Error) charts.

**Validation**
    Use proper validation techniques including cross-validation and external test sets to assess model performance and avoid overfitting.

Principal Component Analysis (PCA)
==================================

Mathematical Background
-----------------------

Principal Component Analysis decomposes a data matrix **X** (N × K) into scores **T** (N × A) and loadings **P** (K × A):

.. math::

    \mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}

where:
    * **T** are the scores (projections of observations onto principal components)
    * **P** are the loadings (directions of maximum variance)
    * **E** is the residual matrix
    * A is the number of components retained

The principal components are ordered by the amount of variance they explain, with the first component explaining the most variance.

PCA Class Documentation
-----------------------

.. autoclass:: process_improve.multivariate.methods.PCA
    :members:
    :inherited-members:
    :show-inheritance:

Parameters and Attributes
-------------------------

**Key Parameters:**

* ``n_components``: Number of principal components to compute
* ``missing_data_settings``: Dictionary for handling missing data (optional)

**Important Attributes:**

* ``x_scores``: Scores matrix (N × A)
* ``x_loadings``: Loadings matrix (K × A) 
* ``explained_variance_``: Variance explained by each component
* ``squared_prediction_error``: SPE values for process monitoring
* ``hotellings_t2``: Hotelling's T² statistics
* ``R2cum``: Cumulative R² values

Basic PCA Example
-----------------

Here's a simple example demonstrating PCA usage with synthetic data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from process_improve.multivariate.methods import PCA, center, scale

    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    # Create correlated data
    base_data = np.random.randn(n_samples, 2)
    X = np.column_stack([
        base_data[:, 0] + 0.5 * base_data[:, 1] + 0.1 * np.random.randn(n_samples),
        base_data[:, 1] + 0.1 * np.random.randn(n_samples),
        0.8 * base_data[:, 0] + 0.2 * np.random.randn(n_samples),
        base_data[:, 0] - base_data[:, 1] + 0.15 * np.random.randn(n_samples),
        0.3 * base_data[:, 1] + 0.2 * np.random.randn(n_samples)
    ])
    
    # Convert to DataFrame with meaningful column names
    X_df = pd.DataFrame(X, columns=['Temperature', 'Pressure', 'Flow_Rate', 'Concentration', 'pH'])
    
    # Center and scale the data
    X_centered = center(X_df)
    X_scaled = scale(X_centered, func=np.std, ddof=1)
    
    # Fit PCA model
    pca_model = PCA(n_components=3)
    pca_model.fit(X_scaled)
    
    # Display results
    print("Explained Variance Ratio:")
    print(pca_model.explained_variance_ratio_)
    print(f"\\nCumulative R²: {pca_model.R2cum.iloc[-1]:.3f}")
    
    # Plot scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score plot
    ax1.scatter(pca_model.x_scores.iloc[:, 0], pca_model.x_scores.iloc[:, 1], alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('PCA Score Plot')
    ax1.grid(True, alpha=0.3)
    
    # Loading plot  
    for i, var in enumerate(X_df.columns):
        ax2.arrow(0, 0, pca_model.x_loadings.iloc[i, 0], pca_model.x_loadings.iloc[i, 1], 
                 head_width=0.02, head_length=0.02, fc='red', ec='red')
        ax2.text(pca_model.x_loadings.iloc[i, 0]*1.1, pca_model.x_loadings.iloc[i, 1]*1.1, 
                var, fontsize=10, ha='center')
    
    ax2.set_xlabel('PC1 Loading')
    ax2.set_ylabel('PC2 Loading') 
    ax2.set_title('PCA Loading Plot')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

Missing Data Handling Example
-----------------------------

The PCA implementation supports several algorithms for handling missing data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from process_improve.multivariate.methods import PCA

    # Create data with missing values
    np.random.seed(42)
    X = np.random.randn(50, 4)
    X_df = pd.DataFrame(X, columns=['Var1', 'Var2', 'Var3', 'Var4'])
    
    # Introduce missing values (10% missing)
    missing_mask = np.random.random(X_df.shape) < 0.1
    X_df_missing = X_df.copy()
    X_df_missing[missing_mask] = np.nan
    
    print(f"Missing values: {X_df_missing.isnull().sum().sum()}")
    
    # Configure missing data settings
    missing_data_settings = {
        'md_method': 'tsr',  # Trimmed Scores Regression
        'md_tol': 1e-6,      # Convergence tolerance
        'md_max_iter': 100   # Maximum iterations
    }
    
    # Fit PCA with missing data
    pca_missing = PCA(n_components=2, missing_data_settings=missing_data_settings)
    pca_missing.fit(X_df_missing)
    
    print(f"Converged in {pca_missing.extra_info['iterations']} iterations")
    print(f"Final convergence error: {pca_missing.extra_info['final_error']:.2e}")

Process Monitoring with PCA
---------------------------

PCA is widely used for process monitoring through statistical control charts:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from process_improve.multivariate.methods import PCA, hotellings_t2_limit, spe_limit

    # Generate training data (normal operation)
    np.random.seed(42)
    X_normal = np.random.randn(100, 6)
    
    # Fit PCA model
    pca_monitor = PCA(n_components=3)
    pca_monitor.fit(X_normal)
    
    # Generate new data with some outliers
    X_new = np.random.randn(30, 6)
    X_new[10:15] += 3  # Introduce outliers
    
    # Predict on new data
    prediction = pca_monitor.predict(X_new)
    
    # Calculate control limits
    t2_limit = pca_monitor.hotellings_t2_limit(conf_level=0.95)
    spe_limit_val = pca_monitor.spe_limit(conf_level=0.95)
    
    # Create monitoring charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Hotelling's T² chart
    ax1.plot(range(len(prediction.hotellings_t2)), prediction.hotellings_t2, 'bo-', markersize=4)
    ax1.axhline(y=t2_limit, color='r', linestyle='--', label=f'T² limit ({t2_limit:.2f})')
    ax1.set_ylabel("Hotelling's T²")
    ax1.set_title("Hotelling's T² Control Chart")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SPE chart
    ax2.plot(range(len(prediction.squared_prediction_error)), prediction.squared_prediction_error, 'go-', markersize=4)
    ax2.axhline(y=spe_limit_val, color='r', linestyle='--', label=f'SPE limit ({spe_limit_val:.2f})')
    ax2.set_xlabel('Observation')
    ax2.set_ylabel('SPE')
    ax2.set_title('Squared Prediction Error Control Chart')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Identify outliers
    outliers_t2 = prediction.hotellings_t2 > t2_limit
    outliers_spe = prediction.squared_prediction_error > spe_limit_val
    
    print(f"Outliers detected by T²: {np.sum(outliers_t2)}")
    print(f"Outliers detected by SPE: {np.sum(outliers_spe)}")

Partial Least Squares (PLS)
===========================

Mathematical Background
-----------------------

Partial Least Squares finds latent variables that maximize the covariance between X and Y matrices. The algorithm decomposes both X and Y:

.. math::

    \mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}_X

    \mathbf{Y} = \mathbf{U}\mathbf{Q}^T + \mathbf{E}_Y

where:
    * **T** are the X-scores
    * **U** are the Y-scores  
    * **P** are the X-loadings
    * **Q** are the Y-loadings
    * **E_X**, **E_Y** are residual matrices

The relationship between X and Y scores is: **U** = **T****B** + **F**, where **B** is the inner relationship.

PLS vs PCA Comparison
--------------------

**PLS Advantages:**
    * Explicitly models X-Y relationships
    * Better predictive performance when Y is available
    * Handles multicollinearity effectively
    * Focuses on variance relevant to prediction

**PCA Advantages:** 
    * Simpler unsupervised approach
    * Better for pure data exploration
    * More interpretable principal components
    * Better for process monitoring applications

PLS Class Documentation
-----------------------

.. autoclass:: process_improve.multivariate.methods.PLS
    :members:
    :inherited-members:
    :show-inheritance:

Basic PLS Example
-----------------

Here's a comprehensive example showing PLS regression:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    from process_improve.multivariate.methods import PLS

    # Generate synthetic process data
    np.random.seed(42)
    n_samples = 150
    
    # Create correlated X variables (process conditions)
    X = np.random.randn(n_samples, 8)
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)  # Temperature correlation
    X[:, 2] = -0.7 * X[:, 0] + 0.3 * X[:, 1] + 0.4 * np.random.randn(n_samples)  # Pressure
    X[:, 3] = 0.6 * X[:, 1] + 0.3 * np.random.randn(n_samples)  # Flow rate
    X[:, 4:] = 0.2 * np.random.randn(n_samples, 4)  # Additional noise variables
    
    # Create Y variables (quality responses) with known relationships
    Y = np.zeros((n_samples, 2))
    Y[:, 0] = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples)  # Quality 1
    Y[:, 1] = -X[:, 0] + 0.8 * X[:, 2] + X[:, 3] + 0.3 * np.random.randn(n_samples)      # Quality 2
    
    # Convert to DataFrames
    X_names = ['Temp', 'Pressure', 'Flow', 'Catalyst', 'pH', 'Noise1', 'Noise2', 'Noise3']
    Y_names = ['Quality_1', 'Quality_2']
    
    X_df = pd.DataFrame(X, columns=X_names)
    Y_df = pd.DataFrame(Y, columns=Y_names)
    
    # Split data for validation
    train_idx = np.arange(100)
    test_idx = np.arange(100, 150)
    
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    Y_train, Y_test = Y_df.iloc[train_idx], Y_df.iloc[test_idx]
    
    # Fit PLS model
    pls_model = PLS(n_components=4)
    pls_model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred_train_state = pls_model.predict(X_train)
    Y_pred_test_state = pls_model.predict(X_test)
    
    # Extract predictions from state objects
    Y_pred_train = Y_pred_train_state.y_hat
    Y_pred_test = Y_pred_test_state.y_hat
    
    # Calculate performance metrics
    r2_train_1 = r2_score(Y_train.iloc[:, 0], Y_pred_train.iloc[:, 0])
    r2_test_1 = r2_score(Y_test.iloc[:, 0], Y_pred_test.iloc[:, 0])
    rmse_test_1 = np.sqrt(mean_squared_error(Y_test.iloc[:, 0], Y_pred_test.iloc[:, 0]))
    
    print(f"Quality 1 - Train R²: {r2_train_1:.3f}")
    print(f"Quality 1 - Test R²: {r2_test_1:.3f}")
    print(f"Quality 1 - Test RMSE: {rmse_test_1:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Predicted vs Actual plots
    axes[0, 0].scatter(Y_train.iloc[:, 0], Y_pred_train.iloc[:, 0], alpha=0.7, label='Training')
    axes[0, 0].scatter(Y_test.iloc[:, 0], Y_pred_test.iloc[:, 0], alpha=0.7, label='Test')
    axes[0, 0].plot([Y_df.iloc[:, 0].min(), Y_df.iloc[:, 0].max()], 
                    [Y_df.iloc[:, 0].min(), Y_df.iloc[:, 0].max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Actual Quality 1')
    axes[0, 0].set_ylabel('Predicted Quality 1')
    axes[0, 0].set_title(f'Quality 1 Prediction (Test R² = {r2_test_1:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score plot
    axes[0, 1].scatter(pls_model.x_scores.iloc[:, 0], pls_model.x_scores.iloc[:, 1], alpha=0.7)
    axes[0, 1].set_xlabel('LV1 Scores')
    axes[0, 1].set_ylabel('LV2 Scores')
    axes[0, 1].set_title('PLS Score Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # X-loadings plot
    for i, var in enumerate(X_names):
        axes[1, 0].arrow(0, 0, pls_model.x_loadings.iloc[i, 0], pls_model.x_loadings.iloc[i, 1],
                        head_width=0.015, head_length=0.015, fc='blue', ec='blue', alpha=0.7)
        axes[1, 0].text(pls_model.x_loadings.iloc[i, 0]*1.1, pls_model.x_loadings.iloc[i, 1]*1.1,
                       var, fontsize=9, ha='center')
    
    axes[1, 0].set_xlabel('LV1 X-Loading')
    axes[1, 0].set_ylabel('LV2 X-Loading')
    axes[1, 0].set_title('PLS X-Loadings Plot')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Y-loadings plot
    for i, var in enumerate(Y_names):
        axes[1, 1].arrow(0, 0, pls_model.y_loadings.iloc[i, 0], pls_model.y_loadings.iloc[i, 1],
                        head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        axes[1, 1].text(pls_model.y_loadings.iloc[i, 0]*1.2, pls_model.y_loadings.iloc[i, 1]*1.2,
                       var, fontsize=10, ha='center')
    
    axes[1, 1].set_xlabel('LV1 Y-Loading')
    axes[1, 1].set_ylabel('LV2 Y-Loading')
    axes[1, 1].set_title('PLS Y-Loadings Plot')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.show()

Cross-Validation for Model Selection
------------------------------------

Proper model selection using cross-validation:

.. code-block:: python

    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from process_improve.multivariate.methods import PLS

    def pls_cross_validation(X, Y, max_components=10, cv_folds=5):
        """Perform cross-validation to select optimal number of PLS components."""
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        n_components_range = range(1, min(max_components + 1, X.shape[1] + 1))
        
        cv_scores = []
        
        for n_comp in n_components_range:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                Y_train_fold, Y_val_fold = Y.iloc[train_idx], Y.iloc[val_idx]
                
                # Fit PLS model
                pls = PLS(n_components=n_comp)
                pls.fit(X_train_fold, Y_train_fold)
                
                # Predict and calculate error
                Y_pred_state = pls.predict(X_val_fold)
                Y_pred = Y_pred_state.y_hat
                mse = mean_squared_error(Y_val_fold, Y_pred)
                fold_scores.append(mse)
            
            cv_scores.append(np.mean(fold_scores))
        
        return n_components_range, cv_scores
    
    # Use the function
    n_comp_range, cv_mse = pls_cross_validation(X_train, Y_train, max_components=6)
    
    # Find optimal number of components
    optimal_n_comp = n_comp_range[np.argmin(cv_mse)]
    
    print(f"Optimal number of components: {optimal_n_comp}")
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    plt.plot(n_comp_range, cv_mse, 'bo-', markersize=8)
    plt.axvline(x=optimal_n_comp, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Number of PLS Components')
    plt.ylabel('Cross-Validation MSE')
    plt.title('PLS Model Selection via Cross-Validation')
    plt.grid(True, alpha=0.3)
    plt.show()

T-shaped Partial Least Squares (TPLS)
=====================================

Mathematical Background and Theory
----------------------------------

T-shaped Partial Least Squares (TPLS) is an advanced multivariate method designed for analyzing complex data structures where observations naturally organize into multiple blocks or batches. This method is particularly valuable in batch process industries, where data collection occurs in structured phases.

The TPLS method handles **T-shaped data** organized into multiple blocks:

* **D block**: Design variables or initial conditions
* **F block**: Process trajectory variables (time-varying)
* **Z block**: Final state variables 
* **Y block**: Quality or response variables

The mathematical formulation extends traditional PLS to handle these interconnected data blocks simultaneously:

.. math::

    \mathbf{D} = \mathbf{T}_D\mathbf{P}_D^T + \mathbf{E}_D

    \mathbf{F} = \mathbf{T}_F\mathbf{P}_F^T + \mathbf{E}_F

    \mathbf{Z} = \mathbf{T}_Z\mathbf{P}_Z^T + \mathbf{E}_Z

    \mathbf{Y} = \mathbf{U}\mathbf{Q}^T + \mathbf{E}_Y

where the latent variables are related through: **U** = **T_D****B_D** + **T_F****B_F** + **T_Z****B_Z** + **F**

TPLS Applications
-----------------

**Batch Process Modeling:**
    * Pharmaceutical manufacturing
    * Chemical reaction optimization
    * Food processing quality control
    * Biotechnology process monitoring

**Multi-Block Data Analysis:**
    * Sensor fusion applications
    * Multi-scale process modeling
    * Quality prediction from multiple data sources
    * Process optimization with multiple objectives

Data Structure Requirements
---------------------------

TPLS requires data organized in a specific **T-shaped** structure:

.. code-block:: python

    # Data organization for TPLS
    from process_improve.multivariate.methods import DataFrameDict
    
    # Example data structure
    data_blocks = DataFrameDict({
        'D': design_dataframe,      # Design conditions (N × K_D)
        'F': trajectory_dataframe,  # Process trajectories (N × K_F) 
        'Z': final_state_dataframe, # Final states (N × K_Z)
        'Y': quality_dataframe      # Quality responses (N × K_Y)
    })

Each block must have the same number of rows (observations/batches) but can have different numbers of columns (variables).

TPLS Class Documentation
------------------------

.. autoclass:: process_improve.multivariate.methods.TPLS
    :members:
    :inherited-members:
    :show-inheritance:

Complete TPLS Example with Synthetic Data
-----------------------------------------

Here's a comprehensive example demonstrating TPLS usage with the correct T-shaped data structure:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from process_improve.multivariate.methods import TPLS, DataFrameDict

    # Generate synthetic T-shaped data following the TPLS structure
    np.random.seed(42)
    n_formulas = 60  # Number of formulations/batches
    n_materials_a, n_materials_b = 8, 6  # Materials in each group
    n_props_a, n_props_b = 4, 3  # Properties for each material group
    n_conditions = 3  # Process conditions
    n_outputs = 2    # Quality outputs
    
    # D block: Material Properties (rows = materials, columns = properties)
    # This describes the intrinsic properties of raw materials
    properties = {
        "Group_A": pd.DataFrame(
            np.random.randn(n_materials_a, n_props_a),
            columns=[f'Density_A', f'Viscosity_A', f'Purity_A', f'Cost_A'],
            index=[f'Material_A{i}' for i in range(n_materials_a)]
        ),
        "Group_B": pd.DataFrame(
            np.random.randn(n_materials_b, n_props_b), 
            columns=[f'Melting_Point_B', f'Hardness_B', f'Color_B'],
            index=[f'Material_B{i}' for i in range(n_materials_b)]
        )
    }
    
    # F block: Formulation Matrix (rows = formulations, columns = materials)
    # Each row represents a formulation recipe
    formulas = {
        "Group_A": pd.DataFrame(
            np.random.exponential(0.2, (n_formulas, n_materials_a)),
            columns=[f'Material_A{i}' for i in range(n_materials_a)],
            index=[f'Formula_{i}' for i in range(n_formulas)]
        ),
        "Group_B": pd.DataFrame(
            np.random.exponential(0.15, (n_formulas, n_materials_b)),
            columns=[f'Material_B{i}' for i in range(n_materials_b)],
            index=[f'Formula_{i}' for i in range(n_formulas)]
        )
    }
    
    # Normalize formulation rows to sum to 1 (percentage composition)
    for group in formulas:
        formulas[group] = formulas[group].div(formulas[group].sum(axis=1), axis=0)
    
    # Z block: Process Conditions (rows = formulations, columns = conditions)
    process_conditions = {
        "Conditions": pd.DataFrame({
            'Temperature': 150 + 50 * np.random.randn(n_formulas),
            'Pressure': 2.0 + 0.5 * np.random.randn(n_formulas), 
            'Time': 120 + 30 * np.random.randn(n_formulas)
        }, index=[f'Formula_{i}' for i in range(n_formulas)])
    }
    
    # Y block: Quality Indicators (rows = formulations, columns = responses)
    # Simulate realistic relationships between inputs and outputs
    temp_effect = (process_conditions["Conditions"]['Temperature'] - 150) / 50
    pressure_effect = (process_conditions["Conditions"]['Pressure'] - 2.0) / 0.5
    
    quality_indicators = {
        "Quality": pd.DataFrame({
            'Strength': 100 + 10 * temp_effect + 5 * pressure_effect + 3 * np.random.randn(n_formulas),
            'Flexibility': 50 - 5 * temp_effect + 8 * pressure_effect + 2 * np.random.randn(n_formulas)
        }, index=[f'Formula_{i}' for i in range(n_formulas)])
    }
    
    # Organize data into the required TPLS structure
    all_data = DataFrameDict({
        "D": properties,      # Material properties
        "F": formulas,        # Formulation recipes  
        "Z": process_conditions,  # Process conditions
        "Y": quality_indicators   # Quality responses
    })
    
    print("TPLS Data Structure:")
    print(f"D (Properties): {len(properties)} groups")
    for group, df in properties.items():
        print(f"  {group}: {df.shape} (materials × properties)")
    print(f"F (Formulas): {len(formulas)} groups") 
    for group, df in formulas.items():
        print(f"  {group}: {df.shape} (formulas × materials)")
    print(f"Z (Conditions): {process_conditions['Conditions'].shape} (formulas × conditions)")
    print(f"Y (Quality): {quality_indicators['Quality'].shape} (formulas × responses)")
    
    # Split data for training and testing
    train_idx = slice(0, 45)  # First 45 formulations for training
    test_idx = slice(45, 60)  # Last 15 for testing
    
    # Create training dataset
    train_data = DataFrameDict({
        "D": properties,  # Properties don't change
        "F": {group: df.iloc[train_idx] for group, df in formulas.items()},
        "Z": {group: df.iloc[train_idx] for group, df in process_conditions.items()},
        "Y": {group: df.iloc[train_idx] for group, df in quality_indicators.items()}
    })
    
    # Fit TPLS model (d_matrix contains the material properties)
    tpls_model = TPLS(n_components=3, d_matrix=properties)
    tpls_model.fit(train_data)
    
    print(f"\\nTPLS model fitted successfully:")
    print(f"  Components: {tpls_model.n_components}")
    print(f"  Samples: {tpls_model.n_samples}")
    print(f"  Training completed successfully")
    
Model Interpretation and Basic Usage
------------------------------------

Understanding TPLS model structure and basic usage patterns:

.. code-block:: python

    # Model interpretation
    print("\\n=== TPLS Model Information ===")
    print(f"Number of components: {tpls_model.n_components}")
    print(f"Number of training samples: {tpls_model.n_samples}")
    print(f"Material groups: {list(properties.keys())}")
    print(f"Is fitted: {tpls_model.is_fitted_}")
    
    # Display data structure summary
    print("\\n=== Data Structure Summary ===")
    total_materials = sum(df.shape[0] for df in properties.values())
    total_properties = sum(df.shape[1] for df in properties.values())
    print(f"Total materials: {total_materials}")
    print(f"Total properties: {total_properties}")
    print(f"Process conditions: {process_conditions['Conditions'].shape[1]}")
    print(f"Quality indicators: {quality_indicators['Quality'].shape[1]}")

Important Notes for TPLS Usage
------------------------------

**Data Structure Requirements:**

1. **D block (Properties)**: Dictionary of DataFrames, where each DataFrame contains material properties (rows = materials, columns = properties)

2. **F block (Formulas)**: Dictionary of DataFrames, where each DataFrame contains formulation recipes (rows = formulations, columns = materials)

3. **Z block (Conditions)**: Dictionary of DataFrames containing process conditions (rows = formulations, columns = conditions)

4. **Y block (Quality)**: Dictionary of DataFrames containing quality responses (rows = formulations, columns = responses)

**Key Requirements:**

* All F, Z, and Y blocks must have the same number of rows (formulations)
* Column names in F blocks must match the index names in corresponding D blocks
* The d_matrix parameter in TPLS constructor should be the D block (properties)
* Use DataFrameDict to organize the data blocks

Utility Functions Documentation
==============================

The multivariate module includes several important utility functions for data preprocessing, statistical calculations, and plotting support.

Data Preprocessing Functions
----------------------------

center() Function
^^^^^^^^^^^^^^^^^

.. autofunction:: process_improve.multivariate.methods.center

**Example Usage:**

.. code-block:: python

    import numpy as np
    import pandas as pd
    from process_improve.multivariate.methods import center

    # Create sample data
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    # Center the data (subtract column means)
    centered_data, means = center(data, extra_output=True)
    print("Original means:", means)
    print("Centered data means:", centered_data.mean())

scale() Function  
^^^^^^^^^^^^^^^^

.. autofunction:: process_improve.multivariate.methods.scale

**Example Usage:**

.. code-block:: python

    import numpy as np
    import pandas as pd
    from process_improve.multivariate.methods import scale

    # Scale to unit variance
    scaled_data, scales = scale(centered_data, func=np.std, ddof=1, extra_output=True)
    print("Scaling factors:", scales)
    print("Scaled data standard deviations:", scaled_data.std(ddof=1))

MCUVScaler Class
^^^^^^^^^^^^^^^^

.. autoclass:: process_improve.multivariate.methods.MCUVScaler
    :members:

**Example Usage:**

.. code-block:: python

    from process_improve.multivariate.methods import MCUVScaler
    import pandas as pd
    import numpy as np

    # Create sample data
    data = pd.DataFrame(np.random.randn(100, 5))
    
    # Create and use scaler
    scaler = MCUVScaler()
    scaler.fit(data)
    
    # Transform data to mean-centered, unit variance
    scaled_data = scaler.transform(data)
    
    # Verify scaling
    print("Means after scaling:", scaled_data.mean().round(10))
    print("Standard deviations after scaling:", scaled_data.std(ddof=1).round(10))
    
    # Inverse transform
    original_data = scaler.inverse_transform(scaled_data)
    print("Data recovery successful:", np.allclose(data, original_data))

Statistical Functions
---------------------

hotellings_t2_limit() Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: process_improve.multivariate.methods.hotellings_t2_limit

**Example Usage:**

.. code-block:: python

    from process_improve.multivariate.methods import hotellings_t2_limit

    # Calculate T² control limits
    limit_95 = hotellings_t2_limit(conf_level=0.95, n_components=3, n_rows=100)
    limit_99 = hotellings_t2_limit(conf_level=0.99, n_components=3, n_rows=100)
    
    print(f"95% confidence limit: {limit_95:.3f}")
    print(f"99% confidence limit: {limit_99:.3f}")

spe_limit() Function
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: process_improve.multivariate.methods.spe_limit

**Example Usage:**

.. code-block:: python

    from process_improve.multivariate.methods import spe_limit

    # Assuming you have a fitted PCA model
    # spe_limit_95 = spe_limit(pca_model, conf_level=0.95)
    # print(f"SPE 95% limit: {spe_limit_95:.3f}")

ellipse_coordinates() Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: process_improve.multivariate.methods.ellipse_coordinates

**Example Usage:**

.. code-block:: python

    from process_improve.multivariate.methods import ellipse_coordinates
    import matplotlib.pyplot as plt
    import pandas as pd

    # Generate ellipse coordinates for confidence region
    scaling_factors = pd.Series([1.0, 1.2])  # Example scaling factors
    
    x_coords, y_coords = ellipse_coordinates(
        score_horiz=0, score_vert=1, 
        conf_level=0.95, n_points=100,
        n_components=2, scaling_factor_for_scores=scaling_factors, n_rows=50
    )
    
    # Plot confidence ellipse
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, 'r-', linewidth=2, label='95% Confidence')
    plt.xlabel('PC1 Scores')
    plt.ylabel('PC2 Scores')
    plt.title('Hotelling T² Confidence Ellipse')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.show()

Performance and Best Practices
==============================

Computational Complexity
-------------------------

**PCA Complexity:**
    * Time: O(min(N·K², K·N²)) for SVD decomposition
    * Space: O(N·K) for data storage plus O(K·A) for loadings
    * Missing data algorithms add iterative overhead: O(iterations × N·K·A)

**PLS Complexity:**
    * Time: O(A·N·K·M) where M is number of Y variables
    * Space: O(N·K + N·M + K·A + M·A)
    * Generally faster than PCA for same number of components

**TPLS Complexity:**
    * Time: O(A·N·(K_D + K_F + K_Z)·M) 
    * Space: O(N·(K_D + K_F + K_Z + M))
    * More complex due to multi-block structure

Memory Usage Considerations
---------------------------

**Large Datasets:**

.. code-block:: python

    # For large datasets, consider chunking or incremental approaches
    import numpy as np
    from process_improve.multivariate.methods import PCA

    def chunk_pca(data, chunk_size=1000, n_components=5):
        """Process large datasets in chunks."""
        n_samples = len(data)
        
        if n_samples <= chunk_size:
            # Small enough to process directly
            pca = PCA(n_components=n_components)
            return pca.fit(data)
        else:
            # For very large datasets, consider using sklearn's IncrementalPCA
            # or subsample the data for initial model building
            subsample_idx = np.random.choice(n_samples, size=min(chunk_size, n_samples), replace=False)
            pca = PCA(n_components=n_components)
            return pca.fit(data.iloc[subsample_idx])

**Memory Optimization Tips:**

1. Use ``float32`` instead of ``float64`` when precision allows
2. Remove unnecessary intermediate variables
3. Use sparse matrices when applicable
4. Consider dimensionality reduction before fitting complex models

Tips for Large Datasets
-----------------------

**Data Preprocessing:**

.. code-block:: python

    # Efficient preprocessing for large datasets
    def efficient_preprocessing(data, center_cols=True, scale_cols=True):
        """Memory-efficient preprocessing."""
        
        if center_cols:
            # Compute means without creating full centered matrix
            means = data.mean()
            data = data - means
        
        if scale_cols:
            # Compute standard deviations
            stds = data.std(ddof=1)
            stds[stds == 0] = 1.0  # Avoid division by zero
            data = data / stds
            
        return data
    
    # Use chunked processing for very large files
    def process_large_csv(filename, chunk_size=10000):
        """Process large CSV files in chunks."""
        chunk_results = []
        
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            # Process each chunk
            processed_chunk = efficient_preprocessing(chunk)
            chunk_results.append(processed_chunk)
        
        return pd.concat(chunk_results, ignore_index=True)

**Model Selection Strategies:**

.. code-block:: python

    # Efficient model selection for large datasets
    def fast_model_selection(X, Y, max_components=10, subsample_size=1000):
        """Fast model selection using subsampling."""
        
        n_samples = len(X)
        
        if n_samples > subsample_size:
            # Subsample for model selection
            idx = np.random.choice(n_samples, size=subsample_size, replace=False)
            X_sub, Y_sub = X.iloc[idx], Y.iloc[idx]
        else:
            X_sub, Y_sub = X, Y
        
        # Perform cross-validation on subsample
        n_comp_range, cv_scores = pls_cross_validation(X_sub, Y_sub, max_components)
        optimal_n_comp = n_comp_range[np.argmin(cv_scores)]
        
        # Fit final model on full dataset
        final_model = PLS(n_components=optimal_n_comp)
        final_model.fit(X, Y)
        
        return final_model, optimal_n_comp

Preprocessing Recommendations
-----------------------------

**General Guidelines:**

1. **Always examine your data first:**

.. code-block:: python

    # Data exploration before modeling
    def data_exploration(data):
        """Comprehensive data exploration."""
        print("=== Data Exploration ===")
        print(f"Shape: {data.shape}")
        print(f"Data types:\\n{data.dtypes}")
        print(f"\\nMissing values:\\n{data.isnull().sum()}")
        print(f"\\nBasic statistics:\\n{data.describe()}")
        
        # Check for constant columns
        constant_cols = data.columns[data.var() == 0]
        if len(constant_cols) > 0:
            print(f"\\nWarning: Constant columns detected: {list(constant_cols)}")
        
        # Check for highly correlated columns
        corr_matrix = data.corr()
        high_corr = np.where((np.abs(corr_matrix) > 0.95) & (corr_matrix != 1.0))
        if len(high_corr[0]) > 0:
            print("\\nWarning: Highly correlated variable pairs detected")

2. **Handle missing data appropriately:**

.. code-block:: python

    # Missing data strategies
    def handle_missing_data(data, strategy='analyze'):
        """Handle missing data with different strategies."""
        
        missing_pct = (data.isnull().sum() / len(data)) * 100
        
        if strategy == 'analyze':
            print("Missing data analysis:")
            for col, pct in missing_pct.items():
                if pct > 0:
                    print(f"  {col}: {pct:.1f}% missing")
        
        elif strategy == 'remove_high_missing':
            # Remove columns with >20% missing
            high_missing_cols = missing_pct[missing_pct > 20].index
            if len(high_missing_cols) > 0:
                print(f"Removing columns with >20% missing: {list(high_missing_cols)}")
                data = data.drop(columns=high_missing_cols)
        
        elif strategy == 'use_algorithm':
            # Use built-in missing data algorithms
            print("Consider using missing_data_settings in PCA/PLS")
            
        return data

3. **Scale appropriately for your data:**

.. code-block:: python

    # Scaling strategies
    def choose_scaling(data, method='auto'):
        """Choose appropriate scaling method."""
        
        # Check if variables have very different scales
        ranges = data.max() - data.min()
        scale_ratio = ranges.max() / ranges.min()
        
        if method == 'auto':
            if scale_ratio > 100:
                print(f"Large scale differences detected (ratio: {scale_ratio:.1f})")
                print("Recommendation: Use unit variance scaling")
                method = 'unit_variance'
            else:
                print("Scale differences are manageable")
                print("Recommendation: Mean centering may be sufficient")
                method = 'center_only'
        
        if method == 'unit_variance':
            scaler = MCUVScaler()
            return scaler.fit_transform(data)
        elif method == 'center_only':
            return center(data)
        else:
            return data

Common Pitfalls and Troubleshooting
===================================

**Common Issues and Solutions:**

1. **Overfitting:**

.. code-block:: python

    # Detect and prevent overfitting
    def check_overfitting(model, X_train, Y_train, X_test, Y_test):
        """Check for overfitting in PLS models."""
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(Y_train, train_pred)
        test_r2 = r2_score(Y_test, test_pred)
        
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Difference: {train_r2 - test_r2:.3f}")
        
        if train_r2 - test_r2 > 0.2:
            print("WARNING: Possible overfitting detected!")
            print("Consider: Reducing n_components, increasing training data, or using regularization")

2. **Poor Model Performance:**

.. code-block:: python

    # Diagnose poor performance
    def diagnose_poor_performance(model, X, Y):
        """Diagnose reasons for poor model performance."""
        
        print("=== Performance Diagnosis ===")
        
        # Check data quality
        print(f"Data shape: {X.shape}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        
        # Check for low variance variables
        low_var_cols = X.columns[X.var() < 1e-10]
        if len(low_var_cols) > 0:
            print(f"Low variance columns: {list(low_var_cols)}")
        
        # Check Y variable distribution
        if hasattr(Y, 'describe'):
            print(f"Y statistics:\\n{Y.describe()}")
        
        # Check correlation between X and Y
        if X.shape[1] < 20:  # Only for reasonable number of variables
            xy_corr = pd.concat([X, Y], axis=1).corr()
            y_corr = xy_corr.iloc[-Y.shape[1]:, :-Y.shape[1]]
            max_corr = np.abs(y_corr).max().max()
            print(f"Maximum |correlation| between X and Y: {max_corr:.3f}")
            
            if max_corr < 0.1:
                print("WARNING: Very low correlation between X and Y variables")

3. **Numerical Issues:**

.. code-block:: python

    # Handle numerical issues
    def check_numerical_issues(data):
        """Check for common numerical issues."""
        
        print("=== Numerical Issues Check ===")
        
        # Check for infinite values
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"WARNING: {inf_count} infinite values detected")
        
        # Check for very large/small values
        numeric_data = data.select_dtypes(include=[np.number])
        max_val = numeric_data.max().max()
        min_val = numeric_data.min().min()
        
        if max_val > 1e10 or min_val < -1e10:
            print(f"WARNING: Very large values detected (range: {min_val:.2e} to {max_val:.2e})")
            print("Consider: Data transformation or scaling")
        
        # Check condition number (for square matrices)
        try:
            corr_matrix = numeric_data.corr()
            if corr_matrix.shape[0] == corr_matrix.shape[1]:
                cond_num = np.linalg.cond(corr_matrix)
                if cond_num > 1e12:
                    print(f"WARNING: High condition number ({cond_num:.2e})")
                    print("Data may be ill-conditioned")
        except:
            pass

**Error Messages and Solutions:**

+-----------------------------------+----------------------------------------+
| Error Message                     | Solution                               |
+===================================+========================================+
| "Singular matrix"                 | Check for constant columns, perfect   |
|                                   | correlations, or insufficient data     |
+-----------------------------------+----------------------------------------+
| "Unable to converge"              | Increase ``md_max_iter`` or            |
|                                   | ``md_tol`` in missing data settings   |
+-----------------------------------+----------------------------------------+
| "Shape mismatch"                  | Verify all data blocks have same      |
|                                   | number of rows (observations)         |
+-----------------------------------+----------------------------------------+
| "Memory error"                    | Reduce dataset size, use chunking,    |
|                                   | or consider incremental methods       |
+-----------------------------------+----------------------------------------+

This comprehensive documentation provides both theoretical background and practical guidance for using the multivariate analysis methods in the process-improve package. The working examples can be copied and adapted for your specific datasets and applications.