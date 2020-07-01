# Spectral Residual Anomaly Detection Module

This folder specifies the Spectral Residual Anomaly Detection module that can be used in Azure Machine Learning designer. The details of the Spectral Residual algorithm can be found at https://arxiv.org/pdf/1906.03821.pdf.

## How to install
1. Install [Azure PowerShell](https://docs.microsoft.com/en-us/powershell/azure/install-az-ps?view=azps-3.8.0) if you don't have one.
2. Install the azure-cli-ml extension
```
# Uninstall azure-cli-ml (the `az ml` commands)
az extension remove -n azure-cli-ml

# Install local version of azure-cli-ml (which includes `az ml module` commands)
az extension add --source https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/13766063/azure_cli_ml-0.1.0.13766063-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/CLI-SDK-Runners-Validation/13766063 --yes
```

3. Prepare the environment
```
# Login
az login
# Show account list, verify your default subscription
az account list --output table
# Set your default subscription if needed
az account set -s "Your subscription name"

# Configure workspace name and resource name
# NOTE: This will set workspace setting only to the current folder. If you change to another folder, you need to set this again.
az ml folder attach -w "Your workspace name" -g "Your resource group name"

# Set default namespace of module to avoid specifying to each of the following commands
az configure --defaults module_namespace=microsoft.com/office
```

4. Set default output format to table to improve experience
```
az configure
...
Do you wish to change your settings? (y/N): y
What default output format would you like?
 [1] json - JSON formatted output that most closely matches API responses.
 [2] jsonc - Colored JSON formatted output that most closely matches API responses.
 [3] table - Human-readable output format.
 [4] tsv - Tab- and Newline-delimited. Great for GREP, AWK, etc.
 [5] yaml - YAML formatted output. An alternative to JSON. Great for configuration files.
 [6] yamlc - Colored YAML formatted output. An alternative to JSON. Great for configuration files.
 [7] none - No output, except for errors and warnings.
Please enter a choice [Default choice(1)]: 3
```

5. Register module
```
az ml module register --spec-file=https://github.com/microsoft/anomalydetector/blob/master/aml_module/module_spec.yaml
```

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

## Output Specification
The output data set will contain a fraction of the following columns according to the `Detect Mode` parameter. If multiple value colums are selected, the result columns will add value column names as postfix.
* `isAnomaly`. The anomaly result.
* `mag`. The magnitude after spectral residual transformation.
* `score`. A value indicates the significance of the anomaly.
In AnomalyAndMargin mode, the following columns will be output in addition the the above three columns.
* `expectedValue`. The expected value of each point.
* `lowerBoundary`. The lower boundary at each point that the algorithm can tolerant as not anomaly.
* `upperBoundary`. The upper boundary at each point that the algorithm can tolerant as not anomaly.
