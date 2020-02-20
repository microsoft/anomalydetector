
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

The project is consisted of three major parts.

1.generate_data.py is used for preprocess the data, where the original continuous time series are splited according to window size and  artificial outliers are injected in proportion. 
`''python
python generate_data.py --data yahoo --window 128
`''
2.train.py is the network trianing module of SR-CNN. SR transformer is applied on each time-series before training.

3.evalue.py is the evaluation module.As mentioned in the paper, we set different delays to verify whether a whole section of anomalies can be detected in time. 
