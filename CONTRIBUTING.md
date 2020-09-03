# Contributing

## Forms of contribution

### Submit an Issue

torch2trt is use case driven.  We originally created it to solve
use cases related to NVIDIA Jetson, but the layer support has grown
largely since it's release and we've found that it has 
helped many other developers as well.  

The growth of torch2trt has been largely driven by issues submitted on [GitHub](https://github.com/NVIDIA-AI-IOT/torch2trt/issues).
We learn a lot from the reported issues. Submitting an issue it is one of the best ways to begin contributing to torch2trt.

The reported issues typically are one of the following,

* A bug or unexpected result
* A model with unsupported layers

If you report an issue, we typically find the following information helpful

* PyTorch version
* TensorRT version
* Platform (ie: Jetson Nano)
* The PyTorch Module you're attempting to convert
* The steps taken to convert the PyTorch module

If you're not sure how to provide any of these pieces of information, don't worry.  Just open the issue
and we're happy to discuss and help work out the details.

### Ask a Question

Another great way to contribute is to ask a question on [GitHub](https://github.com/NVIDIA-AI-IOT/torch2trt/issues).
There are often other developers who share your question, and they may find the discussion helpful.  This also
helps us gauge feature interest and identify gaps in documentation.

### Submit a Pull Request

torch2trt is use case driven and has limited maintainence, for this reason we value community contributions greatly.
Another great way to contribute is by submitting a pull request.  Pull requests which are most likely to be accepted are

* A new converter
* A test case
* A bug fix

If you add a new converter, it is best to include a few test
cases that cross validate the converter against the original PyTorch.  We provide a utility function to do this,
as described in the [Custom Converter](usage/custom_converter.md) usage guide.

Ideally pull requests solve one thing at a time.  This makes it easy
to evaluate the impact that the changes have on the project step-by-step.  The more confident we are that
the changes will not adversely impact the experience of other developers, the more likely we are to accept them.

## Running module test cases

Before any change is accepted, we run the test cases on at least one platform.  This performs a large number
of cross validation checks against PyTorch.  To do this

```bash
python3 -m torch2trt.test --name=converters --tolerance=1e-2
```

This will not hard-fail, but will highlight any build errors or max error checks.  It is helpful if you include
the status of this command in any pull-request, as well as system information like

* PyTorch version
* TensorRT version
* Platform (ie: Jetson Nano)

## Testing documentation

If you have a change that modifies the documentation, it is relatively straightforward to test.  We
use ``mkdocs-material`` for documentation, which parses markdown files in the ``docs`` folder.

To view the docs, simply call

```
./scripts/test_docs.sh
```

And then navigate to ``https://<ip_address>:8000``.

Please note, this will not include dynamically generated documentation pages like the converters page.
These contain cross reference links to the GitHub source code. If you want to test these
you can call 
    
```bash
./scripts/build_docs.sh <github url> <tag>
```
    
Pointing to the public reflection
of your local repository.  For example, if we're working off the upstream master branch, we
would call 
   
```bash
./scripts/build_docs.sh https://github.com/NVIDIA-AI-IOT/torch2trt master
```
    
If your changes are pushed to your fork, you would do 
   
```bash
./scripts/build_docs.sh https://github.com/<user>/torch2trt my_branch
```
    
