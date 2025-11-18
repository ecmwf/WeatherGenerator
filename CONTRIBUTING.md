# WeatherGenerator Contributing Guide 
 
Thank you for your interest in contributing to the WeatherGenerator! We welcome contributions to help build and develop the WeatherGenerator. This guide will help you get started with contributing to our project. 
 
## Table of Contents 
 
1. [Code of Conduct](#code-of-conduct) 
2. [Getting Started](#getting-started) 
3. [How to Contribute](#how-to-contribute) 
   - [Reporting Issues](#reporting-issues) 
   - [Submitting Pull Requests](#submitting-pull-requests) 
4. [Development Guidelines](#development-guidelines) 
   - [Coding Standards](#coding-standards) 
   - [Commit Messages](#commit-messages) 
   - [Testing](#testing) 
5. [Getting Help](#getting-help) 
 
## Code of Conduct 
 
We are committed to fostering a welcoming and inclusive community. By participating in this project, you agree to abide by our [Code of Conduct](CODE-of-CONDUCT.md). 
 
## Getting Started 
 
1. **Fork the repository**: Create a fork of the repository by clicking the "Fork" button at the top right of the repository page. 
2. **Clone your fork**: Clone your fork to your local machine, see (https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). In the terminal, one can use the following command: 
   ```sh 
   git clone https://github.com/your-username/WeatherGenerator.git 
   ``` 
3. **Set up the upstream remote**: Set up the upstream remote to keep your fork up-to-date with the main repository, see again (https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork). In the terminal: 
   ```sh 
   git remote add upstream https://github.com/ecmwf/WeatherGenerator.git 
   ``` 
 
## How to Contribute 
 
### Reporting Issues 
 
If you find a bug or have a feature request, please create an issue on the repository's [issue tracker](https://github.com/ecmwf/WeatherGenerator/issues).  
When reporting an issue, please use the appropriate issue template and provide as much detail as possible, including steps to reproduce the issue and any relevant logs or screenshots. Please take care not to share personal information in the issue tracker (e.g. usernames, passwords, hostnames, etc). Please use the appropriate tags to categorize your issue. 
 
### Submitting Contributions 
 
lease open first an issue on the repository's [issue tracker](https://github.com/ecmwf/WeatherGenerator/issues) that describes the contribution that you are planning. This can be bug fixes or new features. Having a discussion through the issue early on will ensure that your work aligns with the development roadmap for the WeatherGenerator project and that your PR will eventually be accepted. 
 
#### Implementing and Submitting your Contribution 
 
The WeatherGenerator project follows the standard process of pull requests on Github. If you are unfamiliar, consider following the [Github documentaion on pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests). 
Pull requests are expected to have a clear description of the issue. Any significant pull request is expected to have a Github issue associated. 
 
1. **Create a branch**: Create a new branch for your work 
2. **Make your changes**: Make your changes in your feature branch. 
3. **Commit your changes**: Please use clear and descriptive commit messages. 
4. **Push to your fork**: Push your changes to your fork on GitHub 
5. **Open a pull request**: Open a pull request against the `develop` branch of the WeatherGenerator repository. Provide a clear description of your changes and link any relevant issues. 
 
## Development Guidelines 
 
### Coding Standards 
 
Please follow our coding standards to ensure consistency across the codebase. Refer to our [Coding Standards](CODING_STANDARDS.md) document for details on the conventions and best practices we adhere to. 
 
### Commit Messages 
 
Write clear and concise commit messages that describe the changes made. Where possible and relevant, reference any related issues in the commit message. 
 
### Testing 
 
Ensure that your changes are thoroughly tested by: 
1. Running existing tests to ensure that they pass. 
2. Add or update unit tests as necessary to cover your changes.  
3. Make sure that all tests, new and old, pass before submitting a pull request. 

 
Thank you for contributing to the WeatherGenerator! Your contributions are greatly appreciated. 
 
