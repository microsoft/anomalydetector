# Spectral Residual Anomaly Detection Module

This folder specifies the Spectral Residual Anomaly Detection module that can be used in Azure Machine Learning designer. The details of the Spectral Residual algorithm can be found at https://arxiv.org/pdf/1906.03821.pdf.

## How to install
Execute the following command under this folder to register the module in your workspace.
`
az ml module register --spec-file=module_spec.yaml
`

## Input Specification
* `Input`. A data frame directory that contains the data set. The data set should contain at least 12 rows. Each row should contain a timestamp column and one or more columns that are to be detected.
* `Detect Mode`. The following two detect modes are supported.
  1. `AnomalyOnly`. In this mode, the module outputs columns `isAnomaly`, `mag` and `score`.
  2. `AnomalyAndMargin`. In this mode, the module outputs columns `isAnomaly`, `mag`, `score`, `expectedValue`, `lowerBoundary`, `upperBoundary`.
* `Timestamp Column`. The column that contains timestamp. The timestamp should be in ascending order. No duplication is allowed in timestamp.
* `Value Column`. One or more columns that are to be detected. The data in these columns should be numeric. Absolute value greater than 1e100 is not allowed.
* `Batch Size`. The number of rows to be detected in each batch. The batch size should be at least 12. Set this parameter to 0 or negative number if you want to detect all rows in one batch.
* `Threshold`. In AnomalyOnly mode, points are detected as anomaly if its `score` is greater than threshold. In AnomalyAndMargin mode, this parameter and `sensitivity` works together to filter anomaly.
* `Sensitivity`. This parameter is used in AnomalyAndMargin mode to determine the range of the boundaries.
* `Append result column to output`. If this parameter is set, the input data set will be output together with the results. Otherwise, only the results will be output.
* `Compute stats in visualization`. If this parameter is set, the stats of output dataset will be calcualted.

## Output Specification
The output data set will contain a fraction of the following columns according to the `Detect Mode` parameter. If multiple value colums are selected, the result columns will add value column names as postfix.
* `isAnomaly`. The anomaly result.
* `mag`. The magnitude after spectral residual transformation.
* `score`. A value indicates the significance of the anomaly.
In AnomalyAndMargin mode, the following columns will be output in addition the the above three columns.
* `expectedValue`. The expected value of each point.
* `lowerBoundary`. The lower boundary at each point that the algorithm can tolerant as not anomaly.
* `upperBoundary`. The upper boundary at each point that the algorithm can tolerant as not anomaly.
